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
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];

      imports = map (x: localFlake: {perSystem = import x;}) [
        ./assignment0
      ];

      perSystem = {system, ...}: let
        defaultName = "assignment0";
      in {
        packages.default = self.packages.${system}.${defaultName};
        devShells.defaultName = self.packages.${system}.${defaultName};
      };
    };
}
