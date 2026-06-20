## Coding Guidelines
Write clean, concise, idiomatic Python.

Style rules:

- Prefer simple functions over classes unless state or polymorphism is clearly needed.
- Do not add abstractions “for future extensibility” unless requested.
- Avoid unnecessary helper functions, wrapper classes, config objects, factories, registries, and custom exceptions.
- Keep the happy path obvious.
- Use standard library tools before adding dependencies.
- Use type hints for function signatures, but avoid over-engineered typing.
- Prefer list/dict comprehensions only when they stay readable.
- Avoid clever one-liners if they obscure intent.
- Do not catch broad exceptions unless there is a concrete recovery action.
- Do not add logging, retries, CLI parsing, environment handling, or validation unless requested.
- Keep comments sparse. Comment why, not what.
- Remove dead code, unused imports, unused variables, and redundant branches.
- Prefer returning values directly over storing temporary variables used once.
- Make the smallest correct change. Prefer deleting or simplifying existing code over adding new layers.


## Additional Tips
- manage script config with hydra in separate config folder. Avoid using CLI arguments.
- Use Pyright for type check.
- Use ruff to format all src files.
- Prefer direct type constructors like `str(...)`, `int(...)`, `float(...)`, and `bool(...)` over bloated type checks or one-off helper functions. Keep simple conversions simple.
- avoid absolute paths that are not reproducible in other environments. Design the repo to be robust and reproduce-friendly. Most important examples: do not commit paths under `/home/...`, local dataset/cache paths, local checkpoint paths, local Python binaries, or sibling-repo script paths like `../stable-worldmodel/scripts/...` unless they are explicitly documented, configurable, and not required by defaults. Prefer repo-relative paths, Hydra config interpolation, package/module entrypoints, env vars with documented defaults, and manifests that record exact checkpoint/dataset revisions.
- Avoid using default arguments (x: int = 5) and Optional arguments (`| None` included) unless they are absolutely necessary, especially when defining function parameters.
- Avoid capability-probing control flow like nested `hasattr(...)` / fallback `if` branches in core logic. Prefer explicit typed protocols, adapters, or single-purpose helper methods so runtime behavior is clear and unsupported objects fail loudly.
- Prefer explicit task-shaped data contracts over broad `info_dict` filtering, prefix-based key routing, or generic context dictionaries. Pass the few fields a component actually needs.