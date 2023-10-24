pkgs: attrs: with pkgs; with pkgs.python311.pkgs; with attrs; rec {
  version = "0.29.36";

  src = fetchPypi {
    pname = "Cython";
    inherit version;
    hash = "sha256-QcDP0tdU44PJ7rle/8mqSrhH0Ml0cHfd18Dctow7wB8=";
  };

  patches = [];
}
