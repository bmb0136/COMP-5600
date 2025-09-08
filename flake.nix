{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs @ {flake-parts, ...}:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];

      perSystem = inputs @ {pkgs, ...}:
        pkgs.lib.mkMerge (builtins.map (f: (import f) inputs) [
          ./assignment0
        ]);
    };
}
