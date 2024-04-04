#ifndef __READERLP_DEF_HPP__
#define __READERLP_DEF_HPP__

#include <stdexcept>
#include <string>

void inline lpassert(bool condition) {
   if (!condition) {
      throw std::invalid_argument("File not existent or illegal file format.");
   }
}

const std::string LP_KEYWORD_MIN[] = {"minimize", "min", "minimum"};
const std::string LP_KEYWORD_MAX[] = {"maximize", "max", "maximum"};
const std::string LP_KEYWORD_ST[] = {"subject to", "such that", "st", "s.t."};
const std::string LP_KEYWORD_BOUNDS[] = {"bounds", "bound"};
const std::string LP_KEYWORD_INF[] = {"infinity", "inf"};
const std::string LP_KEYWORD_FREE[] = {"free"};
const std::string LP_KEYWORD_GEN[] = {"general", "generals", "gen"};
const std::string LP_KEYWORD_BIN[] = {"binary", "binaries", "bin"};
const std::string LP_KEYWORD_SEMI[] = {"semi-continuous", "semi", "semis"};
const std::string LP_KEYWORD_SOS[] = {"sos"};
const std::string LP_KEYWORD_END[] = {"end"};

const unsigned int LP_KEYWORD_MIN_N = 3;
const unsigned int LP_KEYWORD_MAX_N = 3;
const unsigned int LP_KEYWORD_ST_N = 4;
const unsigned int LP_KEYWORD_BOUNDS_N = 2;
const unsigned int LP_KEYWORD_INF_N = 2;
const unsigned int LP_KEYWORD_FREE_N = 1;
const unsigned int LP_KEYWORD_GEN_N = 3;
const unsigned int LP_KEYWORD_BIN_N = 3;
const unsigned int LP_KEYWORD_SEMI_N = 3;
const unsigned int LP_KEYWORD_SOS_N = 1;
const unsigned int LP_KEYWORD_END_N = 1;

#endif
