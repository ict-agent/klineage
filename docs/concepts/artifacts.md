# Artifact

Artifacts connect the stages of KLineage. A problem describes the computation,
a kernel implements it, and a skill card records an optimization that can be
reused. Each is a tree of files and directories, with descriptions and examples
that explain what belongs in the tree.

<figure class="diagram" tabindex="0" aria-label="An Artifact describes an annotated tree of files and directories.">
  <img src="../assets/diagrams/artifact-tree.svg"
       alt="An Artifact describes an annotated tree of files and directories."
       width="800" height="392" loading="lazy">
</figure>

## Node

`Node` represents either a file or a directory. A directory contains child Nodes;
a file holds text or a `FileRef` pointing to external data. Each Node can also
carry a `description` and a `one_shot` example.

Artifact classes use Node trees to define their expected file layout. In a
layout, a file may omit its contents. In an actual Artifact, every file has
content or a reference, including empty text for an empty file.

Use `check` to compare an actual tree with its layout:

```python
from klineage.artifact import Artifact, Node

layout = Node.directory(".", (
    Node.file("note.txt", description="Plain text.", one_shot="hello"),
))
tree = Node.directory(".", (Node.file("note.txt", "hello"),))
layout.check(tree)
note = Artifact("example.note", tree)
```

The check requires the same names, file/directory types and complete child
structure. It ignores contents, annotations and child order, and reports the
path of a mismatch. Artifact classes apply this check when loading their trees
and add content checks where needed.

Use `find(name)` on a Node or Artifact to search by exact name, including the
root. One match returns a Node, multiple matches return a list, and no match
returns `[]`:

```python
note.find("note.txt").content  # "hello"
note.find("missing.txt")       # []
note.find(".")                 # The root Node
```

Names are not paths. To narrow a search, query a subtree first:
`kernel.find("submission").find("solution")`. If several subtrees share a name,
select the intended Node from the returned list before continuing.

To define another format, subclass Artifact and give it a Node layout. See
[Define an artifact class](../guides/extending.md#define-an-artifact-class).

## Problem Artifact

A `Problem` combines one FlashInfer Trace definition with zero or more workloads.
The definition describes the inputs, outputs, shapes, data types and reference
implementation. Each workload supplies concrete dimensions and input data for
one evaluation case.

Supply a problem as a directory:

```text
problem/
  definition.json     # Computation and reference implementation
  workloads.jsonl     # One workload record per line
  inputs/             # Fixed input files, when used
```

`definition.json` and `workloads.jsonl` are required. Fixed-input paths are
relative to this directory. An empty workload file is valid for a Problem
artifact; generation and evaluation require at least one case.

Workloads share the same definition and have unique UUIDs, so results remain
separate for each case. Kernels reference the original problem directory
throughout optimization.

`Problem.problem_spec()` returns the base Node layout. Problem checks it along
with the fixed-input filenames declared by its workloads:

::: klineage.artifact.Problem.problem_spec
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

Use `load_problem(problem_directory)` to load the directory, or
`klineage.problem(problem_id)` to obtain the Problem from an evaluation service.
See the [Artifact API](../reference/artifacts.md) for signatures.

## CUDA Kernel Artifact

`CudaKernel` combines kernel metadata with the configuration and source code
needed to build an implementation:

```text
current_kernel/
  metadata.json         # Problem reference and evaluation results
  submission/
    config.toml         # Entry function and build settings
    solution/
      kernel.cu         # Host entry and device implementation
```

The source defines the host entry function, device computation and launch
parameters. The problem stays in its original directory and is referenced from
metadata.

`CudaKernel.kernel_spec()` returns the native Node layout. CUDA also supports
an empty submission for generation from scratch and a Python host-entry layout:

::: klineage.backend.cuda.CudaKernel.kernel_spec
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

For a problem named `add_one`, the configuration can be:

```toml
[solution]
name = "add-one"
definition = "add_one"
author = "klineage"

[build]
language = "cuda"
entry_point = "kernel.cu::kernel"
destination_passing_style = true
```

The entry path is relative to `solution/`. With destination-passing style, the
host function receives output tensors after the inputs; the Kernel callable
allocates those outputs and returns them to its caller.

See [CUDA implementation](backends.md#cuda-implementation) for building and timing,
and [Evaluation guide](../guides/evaluation.md) for measuring a candidate.

## Artifact Serialization

Use `klineage.serialization` to exchange Artifacts as JSON or read their file
contents. Shared handlers provide these representations for every Artifact class:

| Interface | Purpose |
| --- | --- |
| `to_json(artifact)` | Serialize the type and annotated file tree |
| `from_json(text, Problem)` | Restore and validate a specific Artifact class |
| `raw_text(value)` | Render an Artifact or Node tree, or read file text |
| `raw_bytes(node_or_reference)` | Read file bytes, including binary data |

For an Artifact or layout, `raw_text` draws tree branches, marks directories with `/`, and
places descriptions after `--->`. Examples appear as
`(See: <path or inline example> as an example)`. Referenced examples keep their
paths so the agent can read them when needed. A file with a payload returns its
contents instead. Use `to_json` when a JSON representation is needed.

The note layout above renders as:

```text
./
└── note.txt ---> Plain text. (See: hello as an example)
```
