pkgs: attrs: with pkgs; with pkgs.python311.pkgs; with attrs; rec {
  version = "3.2.9";

  src = fetchPypi {
    pname = "cython";
    inherit version;
    hash = "sha256-0knJAiqxMoaxe9ZvMGCegAxfle/uywYWiZDHpmzs3mw=";
  };

  patches = [];
}
