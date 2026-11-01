"""Typed artifacts, descriptions, examples and checked file storage."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

from klineage.errors import StructuredOutputError
from klineage.serialization import from_json, raw_bytes, to_json

ROOT_NAME = "."
ACTION_PREFIX = "action."
SHA256_RE = re.compile(r"[0-9a-f]{64}")
TYPE_ID = re.compile(r"[a-z][a-z0-9_.-]*")


class ArtifactKind(StrEnum):
    PROBLEM = "problem"
    KERNEL = "kernel"
    SKILLCARD = "skillcard"
    MEMORY = "memory"
    ACTION = "action.action"
    GENERATION = "action.generation"
    INIT = "action.init"
    DECOMPOSE = "action.decompose"
    APPLY = "action.apply"


ACTION_TYPES = frozenset(
    (ArtifactKind.ACTION, ArtifactKind.GENERATION, ArtifactKind.INIT, ArtifactKind.DECOMPOSE, ArtifactKind.APPLY)
)


class NodeKind(StrEnum):
    DIRECTORY = "directory"
    FILE = "file"


def relative_source_path(value: str, label: str = "source path") -> str:
    """Validate and normalize a portable relative path inside a source tree."""

    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    raw = value
    if not raw or raw != raw.strip():
        raise ValueError(f"{label} must be a normalized relative path")
    if "\x00" in raw:
        raise ValueError(f"{label} must not contain a NUL byte")
    if "\\" in raw:
        raise ValueError(f"{label} must use '/' path separators")
    path = PurePosixPath(raw)
    parts = raw.split("/")
    if (
        path.is_absolute()
        or ":" in parts[0]
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"{label} must be a normalized relative path")
    normalized = path.as_posix()
    if normalized != raw:
        raise ValueError(f"{label} must be a normalized relative path")
    return normalized


def node_name(name: str) -> str:
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("node name must be a nonempty path component")
    if name in (".", "..") or any(char in name for char in ("/", "\\", "\x00")):
        raise ValueError("node name must be a single relative path component")
    return name


@dataclass(frozen=True, slots=True)
class FileRef:
    """A file location pinned by SHA-256 and byte length.

    Construction validates these fields without opening the file. Use
    ``file_ref(path)`` to compute a pin, ``check_ref`` to verify it, or
    ``klineage.serialization.raw_text`` to verify and read UTF-8 content.

    Attributes:
        path: Absolute POSIX file path.
        sha256: Lowercase hexadecimal SHA-256 digest.
        size: Nonnegative file length in bytes.
    """

    path: str
    sha256: str
    size: int

    def __post_init__(self):
        if (
            not isinstance(self.path, str)
            or "\x00" in self.path
            or not PurePosixPath(self.path).is_absolute()
        ):
            raise ValueError("reference needs an absolute file path")
        if not isinstance(self.sha256, str) or not SHA256_RE.fullmatch(self.sha256):
            raise ValueError("reference needs a SHA-256 digest")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("reference size must be nonnegative")


@dataclass(frozen=True, slots=True)
class Node:
    """An immutable annotated directory or file.

    Directories have uniquely named children and no content. Files may contain
    inline text or a ``FileRef`` and have no children. Files without either
    describe expected structure. Names are single path components; only a
    directory may use the root name ``.``.
    Constructing a node does not read its references or examples.

    Attributes:
        name: Relative component, or ``.`` for a tree root.
        kind: Directory or file.
        description: Human-readable constraints or context.
        one_shot: Optional example text or a checked reference to example text.
        children: Ordered directory entries.
        content: Inline UTF-8 file text.
        reference: External file pin, mutually exclusive with ``content``.
        artifact_type: Optional artifact type identifier on a directory node.

    Examples:
        >>> tree = Node.directory(".", (Node.file("note.txt", "hello"),))
        >>> tree.find("note.txt").content
        'hello'
    """

    name: str
    kind: NodeKind
    description: str = ""
    one_shot: str | FileRef | None = None
    children: tuple[Node, ...] = ()
    content: str | None = None
    reference: FileRef | None = None
    artifact_type: str | None = None

    def __post_init__(self):
        if not isinstance(self.kind, NodeKind):
            raise TypeError("node kind must be a NodeKind")
        if self.name != ROOT_NAME or self.kind is not NodeKind.DIRECTORY:
            node_name(self.name)
        if not isinstance(self.description, str):
            raise TypeError("description must be text")
        if self.one_shot is not None and not isinstance(self.one_shot, (str, FileRef)):
            raise TypeError("one_shot must be text or a FileRef")
        object.__setattr__(self, "children", tuple(self.children))
        if self.kind is NodeKind.DIRECTORY:
            if self.content is not None or self.reference is not None:
                raise ValueError("directories contain children, not file contents")
            names = [node_name(child.name) for child in self.children]
            if len(names) != len(set(names)):
                raise ValueError("duplicate child name")
            return
        if self.children or (self.content is not None and self.reference is not None):
            raise ValueError("files cannot have children or both content and reference")
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("inline files contain UTF-8 text")
        if self.reference is not None and not isinstance(self.reference, FileRef):
            raise TypeError("file reference must be a FileRef")

    @classmethod
    def directory(
        cls, name, children=(), *, description="", one_shot=None, artifact_type=None
    ):
        """Create a directory, preserving child order and checking names."""
        return cls(
            name,
            NodeKind.DIRECTORY,
            description,
            one_shot,
            tuple(children),
            artifact_type=artifact_type,
        )

    @classmethod
    def file(cls, name, content=None, *, reference=None, description="", one_shot=None):
        """Create a file, optionally carrying text content or a reference."""
        return cls(
            name,
            NodeKind.FILE,
            description,
            one_shot,
            content=content,
            reference=reference,
        )

    def find(self, name: str) -> Node | list[Node]:
        """Match exact names throughout this tree, including this node.

        Return the node for one match, a list for multiple matches, or an empty
        list for no match. Lists follow preorder and preserve child order.
        Names are not interpreted as paths.

        Raises:
            TypeError: ``name`` is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("node name must be a string")
        matches = []
        pending = [self]
        while pending:
            node = pending.pop()
            if node.name == name:
                matches.append(node)
            pending.extend(reversed(node.children))
        return matches[0] if len(matches) == 1 else matches

    def with_child(self, node: Node) -> Node:
        """Return a directory with this child added or replaced by name."""
        if self.kind is not NodeKind.DIRECTORY:
            raise TypeError("only directories have children")
        children = tuple(
            node if child.name == node.name else child for child in self.children
        )
        if not any(child.name == node.name for child in self.children):
            children += (node,)
        return replace(self, children=children)

    def check(self, rhs: Node) -> None:
        """Require identical names, file/directory kinds and child structure.

        Child order, contents and annotations do not affect the comparison.
        References are never read. Empty directories must match empty directories.

        Raises:
            TypeError: ``rhs`` is not a Node.
            ValueError: A mismatch occurs; the message identifies its path.
        """
        if not isinstance(rhs, Node):
            raise TypeError("tree comparison requires a Node")

        def compare(left, right, path):
            if left.name != right.name:
                raise ValueError(f"{path}: expected name {left.name!r}, got {right.name!r}")
            if left.kind is not right.kind:
                raise ValueError(f"{path}: expected {left.kind}, got {right.kind}")
            children = {child.name: child for child in right.children}
            for child in left.children:
                location = child.name if path == ROOT_NAME else f"{path}/{child.name}"
                other = children.pop(child.name, None)
                if other is None:
                    raise ValueError(f"{location}: missing {child.name}")
                compare(child, other, location)
            if children:
                name = next(iter(children))
                location = name if path == ROOT_NAME else f"{path}/{name}"
                raise ValueError(f"{location}: unexpected {name}")

        compare(self, rhs, self.name)


@dataclass(frozen=True, slots=True)
class Artifact:
    """A typed, immutable file tree with an external JSON representation.

    The root is a directory named ``.``. Construction tags the root with
    ``kind`` and calls ``validate``. Each file needs content or a reference.
    Concrete classes check their Node layouts. JSON carries this tree and its
    annotations; it is not an implicit file in the tree.
    Runtime handles, such as a compiled kernel function, are not serialized.

    Attributes:
        kind: Open-ended type identifier, such as ``problem`` or ``kernel.cuda``.
        tree: Annotated root node.
        artifact_kind: Optional subclass constraint on ``kind``.
        schema: Optional Node layout checked by the default validator.

    Examples:
        >>> item = Artifact("note", Node.directory(".", (Node.file("a.txt", "a"),)))
        >>> from klineage.serialization import from_json, to_json
        >>> from_json(to_json(item), Artifact) == item
        True
    """

    kind: str
    tree: Node
    artifact_kind: ClassVar[str | None] = None
    schema: ClassVar[Node | None] = None

    def __post_init__(self):
        if not isinstance(self.kind, str) or not TYPE_ID.fullmatch(self.kind):
            raise ValueError("artifact needs a type identifier")
        if self.artifact_kind is not None and self.kind != self.artifact_kind:
            raise ValueError("artifact type differs from its class")
        if self.tree.kind is not NodeKind.DIRECTORY or self.tree.name != ROOT_NAME:
            raise ValueError("artifact root must be a '.' directory")
        if self.tree.artifact_type not in (None, self.kind):
            raise ValueError("tree type differs from artifact type")
        pending = [(self.tree, ROOT_NAME)]
        while pending:
            node, path = pending.pop()
            if node.kind is NodeKind.FILE and node.content is None and node.reference is None:
                raise ValueError(f"{path}: artifact file requires content or a reference")
            pending.extend(
                (child, child.name if path == ROOT_NAME else f"{path}/{child.name}")
                for child in node.children
            )
        if self.tree.artifact_type is None:
            object.__setattr__(self, "tree", replace(self.tree, artifact_type=self.kind))
        self.validate()

    def validate(self):
        """Subclass hook for content checks, called during construction.

        Subclasses can call this method to check their schema before validating
        file contents, or check a more specific Node layout directly.
        """
        if self.schema is not None:
            self.schema.check(self.tree)

    def find(self, name: str) -> Node | list[Node]:
        """Find exact names in the tree; return one Node or a list of matches."""
        return self.tree.find(name)

    def part(self, name: str, artifact_type: type[Artifact] | None = None) -> Artifact:
        """Extract one named subtree as an Artifact, optionally checking its class.

        Raises:
            KeyError: No node has this name.
            ValueError: Multiple nodes match, or the node is not a typed artifact.
        """
        node = self.find(name)
        if isinstance(node, list):
            if not node:
                raise KeyError(f"{name}: artifact not found")
            raise ValueError(f"{name}: expected one artifact, found {len(node)} matches")
        if node.artifact_type is None:
            raise ValueError(f"{name} is not a typed artifact")
        return (artifact_type or Artifact).from_tree(node.artifact_type, replace(node, name=ROOT_NAME))

    @classmethod
    def from_tree(cls, kind: str, tree: Node):
        """Restore this class directly, running its validation without building."""
        if cls.artifact_kind is not None and cls.artifact_kind != kind:
            raise TypeError(f"{kind} is not a {cls.__name__}")
        value = object.__new__(cls)
        Artifact.__init__(value, kind, tree)
        return value


HASH_BLOCK_BYTES = 1024 * 1024
MAX_INLINE_BYTES = 1024 * 1024


class FileMode(StrEnum):
    TEXT = "text"
    SOURCE = "source"


def file_ref(path: Path) -> FileRef:
    """Hash a regular file and return its absolute path, digest and byte length.

    The file is read in blocks. Symlinks and non-file paths are rejected.
    """
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"reference must be a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
            size += len(block)
    return FileRef(str(path.resolve()), digest.hexdigest(), size)


def check_ref(reference: FileRef) -> Path:
    """Rehash a referenced file; return its path or reject a changed pin."""
    path = Path(reference.path)
    actual = file_ref(path)
    if (actual.sha256, actual.size) != (reference.sha256, reference.size):
        raise ValueError(f"referenced file changed: {path}")
    return path


def capture(path: Path, mode: FileMode = FileMode.TEXT) -> Node:
    """Capture files once; source mode enforces UTF-8 and bundle size limits."""
    count = size = 0

    def read(entry: Path) -> Node:
        nonlocal count, size
        if entry.is_symlink():
            raise ValueError(f"artifact cannot contain symlinks: {entry}")
        if entry.is_dir():
            return Node.directory(
                entry.name or ROOT_NAME,
                (read(child) for child in sorted(entry.iterdir())),
                description="Captured directory.",
            )
        if not entry.is_file():
            raise ValueError(f"artifact requires regular files: {entry}")

        length = entry.stat().st_size
        if mode is FileMode.SOURCE:
            count += 1
            if count > SOURCE_FILE_LIMIT or size + length > SOURCE_BYTE_LIMIT:
                raise ValueError("source bundle exceeds its file or byte limit")
        if mode is FileMode.SOURCE or length <= MAX_INLINE_BYTES:
            try:
                data = entry.read_bytes()
                content = data.decode("utf-8")
                if "\x00" in content:
                    raise ValueError("text contains a NUL byte")
                size += len(data)
                if mode is FileMode.SOURCE and size > SOURCE_BYTE_LIMIT:
                    raise ValueError("source bundle exceeds its byte limit")
                return Node.file(entry.name, content, description="UTF-8 file.")
            except ValueError:
                if mode is FileMode.SOURCE:
                    raise
        return Node.file(
            entry.name,
            reference=file_ref(entry),
            description="File pinned by SHA-256 and byte length.",
        )

    return read(Path(path))



def save_artifact(artifact: Artifact, path: Path) -> Path:
    """Atomically write an artifact's JSON representation to an explicit file.

    Args:
        artifact: Tree to serialize; external files remain references.
        path: JSON destination, normally outside the artifact's source tree.
            Parent directories are created; an existing file is replaced.

    Returns:
        The absolute destination path.

    Raises:
        IsADirectoryError: The destination is a directory.
        ValueError: The destination is a symlink.
    """
    path = Path(path).expanduser().absolute()
    if path.is_dir():
        raise IsADirectoryError(f"{path}: provide a JSON file path")
    if path.is_symlink():
        raise ValueError("artifact manifest cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(to_json(artifact) + "\n", encoding="utf-8")
        staged.replace(path)
    return path


def load_artifact(path: Path, artifact_type: type[Artifact] = Artifact) -> Artifact:
    """Load JSON or directory files as an explicitly selected Artifact class.

    Args:
        path: JSON file or artifact directory. A directory is captured in full;
            small UTF-8 files are inline and other files use checked references.
        artifact_type: Class that owns validation. Directories require a concrete
            class, or ``Kernel`` to select the backend from the problem reference.

    Returns:
        A validated artifact. Kernel loading here does not compile its callable.
    """
    path = Path(path).expanduser().absolute()
    if path.is_dir():
        from klineage.artifact.kernel import Kernel, capture_kernel

        if issubclass(artifact_type, Kernel):
            tree = capture_kernel(path)
            if artifact_type is Kernel:
                from klineage.artifact.problem import load_problem
                from klineage.backend import get_backend

                problem = load_problem(Artifact(ArtifactKind.KERNEL, tree))
                artifact_type = get_backend(problem.language, problem.platform).kernel_type
        else:
            tree = replace(capture(path), name=ROOT_NAME)
        if artifact_type.artifact_kind is None:
            raise TypeError(f"{path}: a directory requires its Artifact class")
        return artifact_type.from_tree(artifact_type.artifact_kind, tree)
    return from_json(path.read_text(encoding="utf-8"), artifact_type)


SOURCE_BYTE_LIMIT = 16 * 1024 * 1024
SOURCE_FILE_LIMIT = 1024


def read_source_tree(directory: Path, label: str) -> dict[str, str]:
    """Read exact bundle text; reject links, binary files and oversized bundles."""
    directory = Path(directory).expanduser().absolute()
    try:
        if not directory.is_dir():
            raise ValueError("source tree must be a directory")
        files = source_files(capture(directory, FileMode.SOURCE))
        return {
            relative_source_path(name): content
            for name, content in sorted(files.items())
        }
    except (OSError, ValueError) as error:
        raise StructuredOutputError(f"{label}: {error}") from error



def json_node(name: str, value: Any) -> Node:
    """Create an inline JSON file node, ending with a newline."""
    return Node.file(name, to_json(value) + "\n")


def nested(name: str, artifact: Artifact) -> Node:
    """Rename an artifact root for nesting, preserving its type and contents."""
    return replace(artifact.tree, name=name)


def source_tree(sources: Mapping[str, str]) -> Node:
    """Build a root directory from relative POSIX paths and exact source text."""
    entries = {}
    for path, content in sorted(sources.items()):
        parts = relative_source_path(path).split("/")
        current = entries
        for part in parts[:-1]:
            current = current.setdefault(part, {})
            if not isinstance(current, dict):
                raise ValueError("file shadows a source directory")
        if parts[-1] in current:
            raise ValueError("duplicate source path")
        current[parts[-1]] = content

    def build(name, children):
        nodes = tuple(
            build(key, value)
            if isinstance(value, dict)
            else Node.file(key, value, description="Exact UTF-8 source.")
            for key, value in children.items()
        )
        return Node.directory(name, nodes, description="Source directory.")

    return build(".", entries)


def source_files(tree: Node) -> dict[str, str]:
    """Flatten a directory into relative paths and text, checking references."""
    result = {}

    def visit(node, prefix):
        for child in node.children:
            path = f"{prefix}{child.name}"
            if child.kind is NodeKind.DIRECTORY:
                visit(child, path + "/")
                continue
            result[path] = raw_bytes(child).decode("utf-8")

    visit(tree, "")
    return result
