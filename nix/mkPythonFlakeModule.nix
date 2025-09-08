{
  pyproject,
  src,
  extraShellDeps ? pkgs: [],
  extraBuildDeps ? pkgs: [],
  extraDeps ? pkgs: [],
}: {pkgs, ...}: let
  project = (fromTOML (builtins.readFile pyproject)).project;
  dependencies = map (x: pkgs.python3.pkgs.${x}) project.dependencies;
  extra = extraDeps pkgs;
in {
  devShells.${project.name} = pkgs.mkShell {
    packages = dependencies ++ (extraShellDeps pkgs) ++ extra;
  };
  packages.${project.name} = pkgs.python3.pkgs.callPackage ({
    buildPythonPackage,
    setuptools,
    ...
  }:
    buildPythonPackage {
      pname = project.name;
      version = project.version;
      inherit src;
      pyproject = true;
      build-system = [setuptools];
      propagatedBuildInputs = dependencies ++ (extraBuildDeps pkgs) ++ extra;
    }) {};
}
