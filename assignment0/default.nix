{pkgs, ...}: let
  python = pkgs.python3.withPackages (pp:
    with pp; [
      numpy
      matplotlib
    ]);
in {
  devShells.default = pkgs.mkShell {
    packages = [python];
  };
  # packages.default = TODO;
}
