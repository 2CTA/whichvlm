from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path

from cli import app

_LEGACY_MODULES = {
    "cli",
    "constants",
    "data",
    "engine",
    "hardware",
    "models",
    "output",
    "runtime",
    "utils",
}
_LEGACY_PREFIX = f"{__name__}."


def _legacy_real_name(fullname: str) -> str:
    return fullname[len(_LEGACY_PREFIX) :]


class _LegacyImportAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_LEGACY_PREFIX):
            return None

        real_name = _legacy_real_name(fullname)
        if real_name.split(".", 1)[0] not in _LEGACY_MODULES:
            return None

        real_spec = importlib.util.find_spec(real_name)
        if real_spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            self,
            origin=real_spec.origin,
            is_package=real_spec.submodule_search_locations is not None,
        )

    def create_module(self, spec):
        module = importlib.import_module(_legacy_real_name(spec.name))
        sys.modules[spec.name] = module
        return module

    def exec_module(self, module) -> None:
        return None

    def get_code(self, fullname):
        real_name = _legacy_real_name(fullname)
        real_spec = importlib.util.find_spec(real_name)
        if real_spec is None or real_spec.loader is None:
            return None
        return real_spec.loader.get_code(real_name)

    def get_filename(self, fullname):
        real_name = _legacy_real_name(fullname)
        real_spec = importlib.util.find_spec(real_name)
        if real_spec is None or real_spec.origin is None:
            raise ImportError(fullname)
        return real_spec.origin


def _install_legacy_import_aliases() -> None:
    if not any(isinstance(finder, _LegacyImportAlias) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyImportAlias())


_install_legacy_import_aliases()
__path__ = [str(Path(__file__).parent)]


if __name__ == "__main__":
    app()
