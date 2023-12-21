pkgs: attrs: with pkgs; with pkgs.python311.pkgs; with attrs; rec {
  version = "0.29.37";

  src = fetchPypi {
    pname = "Cython";
    inherit version;
    hash = "sha256-+BPUpt2Ure5dT/JmGR0dlb9tQWSk+sxTVCLAIbJQTPs=";
  };

  patches = [];
}
