# Artifact API

Artifacts represent problems, kernels and skills as annotated file trees.
See [Artifact](../concepts/artifacts.md) for the layouts and
[Extension guide](../guides/extending.md#define-an-artifact-class) to define your own.

## Trees and layouts

Use Nodes to describe files and directories, and `check` to compare a tree with
its expected layout. `find(name)` searches recursively by exact name and returns
one Node, a list of matches, or `[]`. FileRefs keep external data referenced by
path and hash.

::: klineage.artifact.FileRef

::: klineage.artifact.Node

::: klineage.artifact.Artifact

## Problems

A Problem holds one definition and its workloads. Load a directory with
`load_problem`, or obtain the registered Problem through `Client.problem()`.

::: klineage.artifact.Problem

::: klineage.artifact.load_problem

::: klineage.artifact.load_trace

## Kernels and results

Load an authored directory with `Kernel.load_kernel()` to build its callable.
Evaluation clients save complete versions under `versions/<N>/`, with measurements
in each version's metadata.

::: klineage.artifact.Kernel
    options:
      members:
        - kernel_spec
        - load_kernel
        - build_kernel
        - validation
        - __call__

::: klineage.artifact.ValidationResult

## Serialization

Shared handlers provide JSON, text and byte representations. Pass the concrete
Artifact class when restoring JSON to apply its validation; runtime callables
are excluded from serialization.

::: klineage.serialization.to_json

::: klineage.serialization.raw_text

::: klineage.serialization.raw_bytes

::: klineage.serialization.from_json

::: klineage.serialization.restore

::: klineage.serialization.SerializationHandler

::: klineage.serialization.register_handler

## File access

Read artifact directories, save JSON and capture source trees with these helpers.

::: klineage.artifact.load_artifact

::: klineage.artifact.save_artifact

::: klineage.artifact.file_ref

::: klineage.artifact.source_tree

::: klineage.artifact.source_files

::: klineage.artifact.nested

::: klineage.artifact.stage_repository

## Tree vocabulary

::: klineage.artifact.ArtifactKind

::: klineage.artifact.NodeKind
