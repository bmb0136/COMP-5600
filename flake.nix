{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs @ {
    self,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} (let
      assignments = [
        ./assignment0
        ./assignment1
        ./assignment2
      ];
    in {
      systems = ["x86_64-linux"];

      imports = map (x: localFlake: {perSystem = import x;}) assignments;

      perSystem = {pkgs, system, ...}: let
        defaultName = builtins.head (builtins.sort (x: y: y < x) (map baseNameOf assignments));
      in {
        packages.default = self.packages.${system}.${defaultName};
        devShells.default = self.packages.${system}.${defaultName};
      };
    });
}
