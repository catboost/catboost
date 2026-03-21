pkgs: attrs: with pkgs; with pkgs.python311.pkgs; with attrs; rec {
  version = "3.1.8";

  src = fetchPypi {
    pname = "cython";
    inherit version;
    hash = "sha256-UYR1++YzujDBV3Zp9BdED0HhojrWtuLbsgcDzUV/BO8=";
  };

  patches = [];
}
