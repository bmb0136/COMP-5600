(import ../nix/mkPythonFlakeModule.nix {
  pyproject = ./pyproject.toml;
  src = ./.;
})
