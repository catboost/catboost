pkgs: attrs: with pkgs; with pkgs.python311.pkgs; with attrs; rec {
  version = "3.0.10";

  src = fetchPypi {
    pname = "Cython";
    inherit version;
    hash = "sha256-3MlnOTMfuFTc9QP5RgdXbP6EiAZsYcpQ39VYNvEy3pk=";
  };

  patches = [];
}
