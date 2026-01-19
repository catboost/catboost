pkgs: attrs: with pkgs; with pkgs.python311.pkgs; with attrs; rec {
  version = "3.1.6";

  src = fetchPypi {
    pname = "cython";
    inherit version;
    hash = "sha256-/0zP/PmPMKtXI/xFo5wFSKP2qxTwHXOTDFv66kVf8Bw=";
  };

  patches = [];
}
