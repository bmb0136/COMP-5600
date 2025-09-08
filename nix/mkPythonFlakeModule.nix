{
  pyproject,
  src,
  extraShellDeps ? pkgs: [],
  extraBuildDeps ? pkgs: [],
  extraDeps ? pkgs: [],
  notebook ? false,
}: {
  pkgs,
  lib,
  ...
}: let
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
  packages."${project.name}-notebook" = lib.mkIf notebook (let
    name = "${project.name}-notebook";
    mainFile = builtins.replaceStrings ["."] ["/"] (builtins.head (lib.strings.splitString ":" project.scripts.${project.name}));
    converter = pkgs.callPackage ../converter {};
  in
    pkgs.stdenv.mkDerivation {
      inherit name src;
      inherit (project) version;
      buildPhase = ''
        mkdir -p $out
        converter ${mainFile}.py $out/${name}.ipynb
        jupyter execute $out/${name}.ipynb
      '';
      nativeBuildInputs = [converter pkgs.jupyter] ++ dependencies;
    });
}
