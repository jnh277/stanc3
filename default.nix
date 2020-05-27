with (import (builtins.fetchTarball {
  name = "nixpkgs-20.03";
  # Tarball of tagged release of Nixpkgs 20.03
  url = "https://github.com/nixos/nixpkgs/archive/5272327b81ed355bbed5659b8d303cf2979b6953.tar.gz";
  # Tarball hash obtained using `nix-prefetch-url --unpack <url>`
  sha256 = "0182ys095dfx02vl2a20j1hz92dx3mfgz2a6fhn31bqlp1wa8hlq";
}) {});

ocamlPackages.buildDunePackage rec {
  pname = "stanc";
  version = "2.23.0";

  src =
    let
      filePrefixesToKeep = builtins.map toString
        [ ./src ./dune-project ./stanc.opam ];
    in
      builtins.path {
        name = "stanc";
        path = ./.;
	# Only depend on necessary files to minimize rebuilds
        filter = (path: type:
          builtins.any
            (prefix:
              lib.strings.hasPrefix prefix path)
            filePrefixesToKeep
        );
      };

  doCheck = false;
  useDune2 = true;

  buildInputs = with ocamlPackages; [
    yojson
    menhir
    core_kernel
    ppx_jane
    ppx_deriving
    findlib
    stdlib-shims
    fmt
    re
  ];

  meta = {
    homepage = https://github.com/stan-dev/stanc3;
    description = "The Stan transpiler (from Stan to C++ and beyond)";
    license = stdenv.lib.licenses.bsd3;
    maintainers = with stdenv.lib.maintainers; [ rybern ];
  };
}
