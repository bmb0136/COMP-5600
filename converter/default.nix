{pkgs, ...}:
(pkgs.callPackage (import ../nix/mkPythonFlakeModule.nix {
  pyproject = ./pyproject.toml;
  src = ./.;
}) {}).packages.converter
