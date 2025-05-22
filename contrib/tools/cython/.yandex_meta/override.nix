pkgs: attrs: with pkgs; with pkgs.python311.pkgs; with attrs; rec {
  version = "3.0.11";

  src = fetchPypi {
    pname = "cython";
    inherit version;
    hash = "sha256-cUbdKvhoK0ymEzGFHmrrzp/lFY51MAND+AwHyoCx+v8=";
  };

  patches = [];
}
