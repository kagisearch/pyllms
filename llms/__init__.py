from __future__ import annotations

import typing as t

from .llms import LLMS

Spec = dict[str, dict[str, t.Any]]
SingleSpec = t.Union[str, tuple[str, dict[str, t.Any]]]
SpecType = t.Union[SingleSpec, list[SingleSpec], Spec]


def as_spec(v: SpecType) -> Spec:
    if isinstance(v, str):
        return {v: {}}
    if isinstance(v, tuple):
        return {v[0]: v[1]}

    if isinstance(v, list):
        if isinstance(v[0], str):
            return {t.cast(str, k): {} for k in v}
        return {k: t.cast(dict[str, t.Any], kw) for k, kw in v}  # noqa: C416
    return v


def init(model: SpecType | None = None, provider: SpecType | None = None, **kwargs) -> LLMS:
    if model:
        spec = as_spec(model)
    elif provider:
        spec = as_spec(provider)
    else:
        return LLMS.default_provider(**kwargs)

    llm = LLMS()
    for k, kw in spec.items():
        llm.add_provider(model=k, **kw)
    return llm


__all__ = ["init", "LLMS"]
