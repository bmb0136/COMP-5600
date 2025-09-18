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
  python = pkgs.python3.withPackages (pp: map (x: pp.${x}) project.dependencies);
  extra = extraDeps pkgs;
in {
  devShells.${project.name} = pkgs.mkShell {
    packages = (extraShellDeps pkgs) ++ extra ++ [python pkgs.jupyter];
  };
  packages.${project.name} = python.pkgs.callPackage ({
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
      propagatedBuildInputs = (extraBuildDeps pkgs) ++ extra ++ [python];
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
      '';
      nativeBuildInputs = [converter];
    });
}
