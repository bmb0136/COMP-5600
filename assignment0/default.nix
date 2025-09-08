(import ../nix/mkPythonFlakeModule.nix {
  pyproject = ./pyproject.toml;
  src = ./.;
  extraShellDeps = pkgs: [pkgs.python3.pkgs.types-matplotib];
})
