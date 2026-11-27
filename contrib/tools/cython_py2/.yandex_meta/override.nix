pkgs: attrs: with pkgs; with pkgs.python311.pkgs; with attrs; rec {
  version = "3.0.12";

  src = fetchPypi {
    pname = "cython";
    inherit version;
    hash = "sha256-uYi7KXznbGceKMl9AXuVQRAQ98d/pmI90LtH7tGu4bw=";
  };

  patches = [];
}
