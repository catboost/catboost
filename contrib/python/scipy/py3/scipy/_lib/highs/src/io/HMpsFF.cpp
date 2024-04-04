/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2022 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/*    Authors: Julian Hall, Ivet Galabova, Leona Gottwald and Michael    */
/*    Feldmeier                                                          */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "io/HMpsFF.h"

#include "lp_data/HighsModelUtils.h"

#ifdef ZLIB_FOUND
#include "zstr.hpp"
#endif

namespace free_format_parser {

FreeFormatParserReturnCode HMpsFF::loadProblem(
    const HighsLogOptions& log_options, const std::string filename,
    HighsModel& model) {
  HighsLp& lp = model.lp_;
  HighsHessian& hessian = model.hessian_;
  FreeFormatParserReturnCode result = parse(log_options, filename);
  if (result != FreeFormatParserReturnCode::kSuccess) return result;

  if (!qrows_entries.empty()) {
    highsLogUser(log_options, HighsLogType::kError,
                 "Quadratic rows not supported by HiGHS\n");
    return FreeFormatParserReturnCode::kParserError;
  }
  if (!sos_entries.empty()) {
    highsLogUser(log_options, HighsLogType::kError,
                 "SOS not supported by HiGHS\n");
    return FreeFormatParserReturnCode::kParserError;
  }
  if (!cone_entries.empty()) {
    highsLogUser(log_options, HighsLogType::kError,
                 "Cones not supported by HiGHS\n");
    return FreeFormatParserReturnCode::kParserError;
  }
  // Duplicate row and column names in MPS files occur if the same row
  // name appears twice in the ROWS section, or if a column name
  // reoccurs in the COLUMNS section after another column has been
  // defined. They are anomalies, but are only handled by a warning in
  // some solvers. Hence, rather than fail, HiGHS does the same.
  //
  // If there are duplicate row (column) names, then they are treated
  // as distinct rows (columns), so the row (column) names array is
  // not valid. Report this for the first instance, and clear the row
  // (column) names array.
  //
  // Note that rowname2idx and colname2idx will return the index
  // corresponding to the first occurrence of the name, so values for
  // rows in the COLUMNS, RHS and RANGES sections, and columns in the
  // BOUNDS and other sections can only be defined for the first
  // occurrence
  if (has_duplicate_row_name_) {
    highsLogUser(log_options, HighsLogType::kWarning,
                 "Linear constraints %d and %d have the same name \"%s\"\n",
                 (int)duplicate_row_name_index0_,
                 (int)duplicate_row_name_index1_, duplicate_row_name_.c_str());
    row_names.clear();
  }
  if (has_duplicate_col_name_) {
    highsLogUser(log_options, HighsLogType::kWarning,
                 "Variables %d and %d have the same name \"%s\"\n",
                 (int)duplicate_col_name_index0_,
                 (int)duplicate_col_name_index1_, duplicate_col_name_.c_str());
    col_names.clear();
  }
  col_cost.assign(num_col, 0);
  for (auto i : coeffobj) col_cost[i.first] = i.second;
  HighsInt status = fillMatrix(log_options);
  if (status) return FreeFormatParserReturnCode::kParserError;
  status = fillHessian(log_options);
  if (status) return FreeFormatParserReturnCode::kParserError;

  lp.num_row_ = num_row;
  lp.num_col_ = num_col;

  lp.sense_ = obj_sense;
  lp.offset_ = obj_offset;

  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = std::move(a_start);
  lp.a_matrix_.index_ = std::move(a_index);
  lp.a_matrix_.value_ = std::move(a_value);
  // a must have at least start_[0]=0 for the fictitious column
  // 0
  if ((int)lp.a_matrix_.start_.size() == 0) lp.a_matrix_.clear();
  lp.col_cost_ = std::move(col_cost);
  lp.col_lower_ = std::move(col_lower);
  lp.col_upper_ = std::move(col_upper);
  lp.row_lower_ = std::move(row_lower);
  lp.row_upper_ = std::move(row_upper);

  lp.objective_name_ = objective_name;
  lp.row_names_ = std::move(row_names);
  lp.col_names_ = std::move(col_names);

  // Only set up lp.integrality_ if non-continuous
  bool is_mip = false;
  for (HighsInt iCol = 0; iCol < (int)col_integrality.size(); iCol++) {
    if (col_integrality[iCol] != HighsVarType::kContinuous) {
      is_mip = true;
      break;
    }
  }
  if (is_mip) lp.integrality_ = std::move(col_integrality);

  hessian.dim_ = q_dim;
  hessian.format_ = HessianFormat::kTriangular;
  hessian.start_ = std::move(q_start);
  hessian.index_ = std::move(q_index);
  hessian.value_ = std::move(q_value);
  // hessian must have at least start_[0]=0 for the fictitious column
  // 0
  if (hessian.start_.size() == 0) hessian.clear();

  // Set the objective name, creating one if necessary
  lp.objective_name_ = findModelObjectiveName(&lp, &hessian);
  lp.cost_row_location_ = cost_row_location;

  return FreeFormatParserReturnCode::kSuccess;
}

HighsInt HMpsFF::fillMatrix(const HighsLogOptions& log_options) {
  HighsInt num_entries = entries.size();
  if (num_entries != num_nz) return 1;

  a_value.resize(num_nz);
  a_index.resize(num_nz);
  a_start.assign(num_col + 1, 0);
  // Nothing to do if there are no entries in the matrix
  if (!num_entries) return 0;

  HighsInt newColIndex = std::get<0>(entries.at(0));

  for (HighsInt k = 0; k < num_nz; k++) {
    a_value.at(k) = std::get<2>(entries.at(k));
    a_index.at(k) = std::get<1>(entries.at(k));

    if (std::get<0>(entries.at(k)) != newColIndex) {
      HighsInt nEmptyCols = std::get<0>(entries.at(k)) - newColIndex;
      newColIndex = std::get<0>(entries.at(k));
      if (newColIndex >= num_col) return 1;

      a_start.at(newColIndex) = k;
      for (HighsInt i = 1; i < nEmptyCols; i++) {
        a_start.at(newColIndex - i) = k;
      }
    }
  }

  for (HighsInt col = newColIndex + 1; col <= num_col; col++)
    a_start[col] = num_nz;

  for (HighsInt i = 0; i < num_col; i++) {
    if (a_start[i] > a_start[i + 1]) {
      highsLogUser(log_options, HighsLogType::kError,
                   "Non-monotonic starts in MPS file reader\n");
      return 1;
    }
  }

  return 0;
}

HighsInt HMpsFF::fillHessian(const HighsLogOptions& log_options) {
  HighsInt num_entries = q_entries.size();
  if (!num_entries) {
    q_dim = 0;
    return 0;
  } else {
    q_dim = num_col;
  }

  q_start.resize(q_dim + 1);
  q_index.resize(num_entries);
  q_value.resize(num_entries);

  // Use q_length to determine the number of entries in each column,
  // and then as workspace to point to the next entry to be filled in
  // each column
  std::vector<HighsInt> q_length;
  q_length.assign(q_dim, 0);

  for (HighsInt iEl = 0; iEl < num_entries; iEl++) {
    HighsInt iCol = std::get<1>(q_entries[iEl]);
    q_length[iCol]++;
  }
  q_start[0] = 0;
  for (HighsInt iCol = 0; iCol < num_col; iCol++) {
    q_start[iCol + 1] = q_start[iCol] + q_length[iCol];
    q_length[iCol] = q_start[iCol];
  }

  for (HighsInt iEl = 0; iEl < num_entries; iEl++) {
    HighsInt iRow = std::get<0>(q_entries[iEl]);
    HighsInt iCol = std::get<1>(q_entries[iEl]);
    double value = std::get<2>(q_entries[iEl]);
    q_index[q_length[iCol]] = iRow;
    q_value[q_length[iCol]] = value;
    q_length[iCol]++;
  }
  return 0;
}

FreeFormatParserReturnCode HMpsFF::parse(const HighsLogOptions& log_options,
                                         const std::string& filename) {
  HMpsFF::Parsekey keyword = HMpsFF::Parsekey::kNone;

  highsLogDev(log_options, HighsLogType::kInfo,
              "readMPS: Trying to open file %s\n", filename.c_str());
#ifdef ZLIB_FOUND
  zstr::ifstream f;
  try {
    f.open(filename.c_str(), std::ios::in);
  } catch (const strict_fstream::Exception& e) {
    highsLogDev(log_options, HighsLogType::kInfo, e.what());
    return FreeFormatParserReturnCode::kFileNotFound;
  }
#else
  std::ifstream f;
  f.open(filename.c_str(), std::ios::in);
#endif
  if (f.is_open()) {
    start_time = getWallTime();
    num_row = 0;
    num_col = 0;
    num_nz = 0;
    cost_row_location = -1;
    // Indicate that no duplicate rows or columns have been found
    has_duplicate_row_name_ = false;
    has_duplicate_col_name_ = false;
    // parsing loop
    while (keyword != HMpsFF::Parsekey::kFail &&
           keyword != HMpsFF::Parsekey::kEnd &&
           keyword != HMpsFF::Parsekey::kTimeout) {
      if (cannotParseSection(log_options, keyword)) {
        f.close();
        return FreeFormatParserReturnCode::kParserError;
      }
      switch (keyword) {
        case HMpsFF::Parsekey::kObjsense:
          keyword = parseObjsense(log_options, f);
          break;
        case HMpsFF::Parsekey::kRows:
          keyword = parseRows(log_options, f);
          break;
        case HMpsFF::Parsekey::kCols:
          keyword = parseCols(log_options, f);
          break;
        case HMpsFF::Parsekey::kRhs:
          keyword = parseRhs(log_options, f);
          break;
        case HMpsFF::Parsekey::kBounds:
          keyword = parseBounds(log_options, f);
          break;
        case HMpsFF::Parsekey::kRanges:
          keyword = parseRanges(log_options, f);
          break;
        case HMpsFF::Parsekey::kQmatrix:
        case HMpsFF::Parsekey::kQuadobj:
          keyword = parseHessian(log_options, f, keyword);
          break;
        case HMpsFF::Parsekey::kQsection:
        case HMpsFF::Parsekey::kQcmatrix:
          keyword = parseQuadRows(log_options, f, keyword);
          break;
        case HMpsFF::Parsekey::kCsection:
          keyword = parseCones(log_options, f);
          break;
        case HMpsFF::Parsekey::kSets:
        case HMpsFF::Parsekey::kSos:
          keyword = parseSos(log_options, f, keyword);
          break;
        case HMpsFF::Parsekey::kFail:
          f.close();
          return FreeFormatParserReturnCode::kParserError;
        case HMpsFF::Parsekey::kFixedFormat:
          f.close();
          return FreeFormatParserReturnCode::kFixedFormat;
        default:
          keyword = parseDefault(log_options, f);
          break;
      }
    }

    // Assign bounds to columns that remain binary by default
    for (HighsInt colidx = 0; colidx < num_col; colidx++) {
      if (col_binary[colidx]) {
        col_lower[colidx] = 0.0;
        col_upper[colidx] = 1.0;
      }
    }

    if (keyword == HMpsFF::Parsekey::kFail) {
      f.close();
      return FreeFormatParserReturnCode::kParserError;
    }
  } else {
    highsLogDev(log_options, HighsLogType::kInfo,
                "readMPS: Not opened file OK\n");
    f.close();
    return FreeFormatParserReturnCode::kFileNotFound;
  }

  f.close();

  if (keyword == HMpsFF::Parsekey::kTimeout)
    return FreeFormatParserReturnCode::kTimeout;

  assert(col_lower.size() == unsigned(num_col));
  assert(row_lower.size() == unsigned(num_row));
  return FreeFormatParserReturnCode::kSuccess;
}

bool HMpsFF::cannotParseSection(const HighsLogOptions& log_options,
                                const HMpsFF::Parsekey keyword) {
  switch (keyword) {
      // Identify the sections that can be parsed
    case HMpsFF::Parsekey::kDelayedrows:
      highsLogUser(log_options, HighsLogType::kError,
                   "MPS file reader cannot parse DELAYEDROWS section\n");
      break;
    case HMpsFF::Parsekey::kModelcuts:
      highsLogUser(log_options, HighsLogType::kError,
                   "MPS file reader cannot parse MODELCUTS section\n");
      break;
    case HMpsFF::Parsekey::kIndicators:
      highsLogUser(log_options, HighsLogType::kError,
                   "MPS file reader cannot parse INDICATORS section\n");
      break;
    case HMpsFF::Parsekey::kGencons:
      highsLogUser(log_options, HighsLogType::kError,
                   "MPS file reader cannot parse GENCONS section\n");
      break;
    case HMpsFF::Parsekey::kPwlobj:
      highsLogUser(log_options, HighsLogType::kError,
                   "MPS file reader cannot parse PWLOBJ section\n");
      break;
    case HMpsFF::Parsekey::kPwlnam:
      highsLogUser(log_options, HighsLogType::kError,
                   "MPS file reader cannot parse PWLNAM section\n");
      break;
    case HMpsFF::Parsekey::kPwlcon:
      highsLogUser(log_options, HighsLogType::kError,
                   "MPS file reader cannot parse PWLCON section\n");
      break;
    default:
      return false;
  }
  return true;
}

// Assuming string is not empty.
HMpsFF::Parsekey HMpsFF::checkFirstWord(std::string& strline, HighsInt& start,
                                        HighsInt& end,
                                        std::string& word) const {
  start = strline.find_first_not_of(" ");
  if ((start == (HighsInt)strline.size() - 1) || is_empty(strline[start + 1])) {
    end = start + 1;
    word = strline[start];
    return HMpsFF::Parsekey::kNone;
  }

  end = first_word_end(strline, start + 1);

  word = strline.substr(start, end - start);

  // store rest of strline for keywords that have arguments
  if (word == "QCMATRIX" || word == "QSECTION" || word == "CSECTION")
    section_args = strline.substr(end, strline.length());

  if (word == "NAME")
    return HMpsFF::Parsekey::kName;
  else if (word == "OBJSENSE")
    return HMpsFF::Parsekey::kObjsense;
  else if (word == "MAX")
    return HMpsFF::Parsekey::kMax;
  else if (word == "MIN")
    return HMpsFF::Parsekey::kMin;
  else if (word == "ROWS")
    return HMpsFF::Parsekey::kRows;
  else if (word == "COLUMNS")
    return HMpsFF::Parsekey::kCols;
  else if (word == "RHS")
    return HMpsFF::Parsekey::kRhs;
  else if (word == "BOUNDS")
    return HMpsFF::Parsekey::kBounds;
  else if (word == "RANGES")
    return HMpsFF::Parsekey::kRanges;
  else if (word == "QSECTION")
    return HMpsFF::Parsekey::kQsection;
  else if (word == "QMATRIX")
    return HMpsFF::Parsekey::kQmatrix;
  else if (word == "QUADOBJ")
    return HMpsFF::Parsekey::kQuadobj;
  else if (word == "QCMATRIX")
    return HMpsFF::Parsekey::kQcmatrix;
  else if (word == "CSECTION")
    return HMpsFF::Parsekey::kCsection;
  else if (word == "DELAYEDROWS")
    return HMpsFF::Parsekey::kDelayedrows;
  else if (word == "MODELCUTS")
    return HMpsFF::Parsekey::kModelcuts;
  else if (word == "INDICATORS")
    return HMpsFF::Parsekey::kIndicators;
  else if (word == "SETS")
    return HMpsFF::Parsekey::kSets;
  else if (word == "SOS")
    return HMpsFF::Parsekey::kSos;
  else if (word == "GENCONS")
    return HMpsFF::Parsekey::kGencons;
  else if (word == "PWLOBJ")
    return HMpsFF::Parsekey::kPwlobj;
  else if (word == "PWLNAM")
    return HMpsFF::Parsekey::kPwlnam;
  else if (word == "PWLCON")
    return HMpsFF::Parsekey::kPwlcon;
  else if (word == "ENDATA")
    return HMpsFF::Parsekey::kEnd;
  else
    return HMpsFF::Parsekey::kNone;
}

HighsInt HMpsFF::getColIdx(const std::string& colname, const bool add_if_new) {
  // look up column name
  auto mit = colname2idx.find(colname);
  if (mit != colname2idx.end()) return mit->second;

  if (!add_if_new) return -1;
  // add new continuous column with default bounds
  colname2idx.emplace(colname, num_col++);
  col_names.push_back(colname);
  col_integrality.push_back(HighsVarType::kContinuous);
  col_binary.push_back(false);
  col_lower.push_back(0.0);
  col_upper.push_back(kHighsInf);

  return num_col - 1;
}

HMpsFF::Parsekey HMpsFF::parseDefault(const HighsLogOptions& log_options,
                                      std::istream& file) {
  std::string strline, word;
  if (getline(file, strline)) {
    strline = trim(strline);
    if (strline.empty()) return HMpsFF::Parsekey::kComment;
    HighsInt s, e;
    HMpsFF::Parsekey key = checkFirstWord(strline, s, e, word);
    if (key == HMpsFF::Parsekey::kName) {
      // Save name of the MPS file
      if (e < (HighsInt)strline.length()) {
        mps_name = first_word(strline, e);
      }
      highsLogDev(log_options, HighsLogType::kInfo,
                  "readMPS: Read NAME    OK\n");
      return HMpsFF::Parsekey::kNone;
    }

    if (key == HMpsFF::Parsekey::kObjsense) {
      // Look for Gurobi-style definition of MAX/MIN on OBJSENSE line
      if (e < (HighsInt)strline.length()) {
        std::string sense = first_word(strline, e);
        if (sense.compare("MAX") == 0) {
          // Found MAX sense on OBJSENSE line
          obj_sense = ObjSense::kMaximize;
        } else if (sense.compare("MIN") == 0) {
          // Found MIN sense on OBJSENSE line
          obj_sense = ObjSense::kMinimize;
        }
        // Don't return HMpsFF::Parsekey::kNone; in case there's a
        // redefinition of OBJSENSE on the "proper" line. If there's
        // no such line, the ROWS keyword is read OK
      }
    }

    return key;
  }
  return HMpsFF::Parsekey::kFail;
}

double getWallTime() {
  using namespace std::chrono;
  return duration_cast<duration<double> >(wall_clock::now().time_since_epoch())
      .count();
}

HMpsFF::Parsekey HMpsFF::parseObjsense(const HighsLogOptions& log_options,
                                       std::istream& file) {
  std::string strline, word;

  while (getline(file, strline)) {
    if (is_empty(strline) || strline[0] == '*') continue;

    HighsInt start = 0;
    HighsInt end = 0;

    HMpsFF::Parsekey key = checkFirstWord(strline, start, end, word);

    // Interpret key being MAX or MIN
    if (key == HMpsFF::Parsekey::kMax) {
      obj_sense = ObjSense::kMaximize;
      continue;
    }
    if (key == HMpsFF::Parsekey::kMin) {
      obj_sense = ObjSense::kMinimize;
      continue;
    }
    highsLogDev(log_options, HighsLogType::kInfo,
                "readMPS: Read OBJSENSE OK\n");
    // start of new section?
    if (key != HMpsFF::Parsekey::kNone) {
      return key;
    }
  }
  return HMpsFF::Parsekey::kFail;
}

HMpsFF::Parsekey HMpsFF::parseRows(const HighsLogOptions& log_options,
                                   std::istream& file) {
  std::string strline, word;
  bool hasobj = false;
  // Assign a default objective name
  objective_name = "Objective";

  assert(num_row == 0);
  assert(row_lower.size() == 0);
  assert(row_upper.size() == 0);
  while (getline(file, strline)) {
    if (is_empty(strline) || strline[0] == '*') continue;
    double current = getWallTime();
    if (time_limit > 0 && current - start_time > time_limit)
      return HMpsFF::Parsekey::kTimeout;

    bool isobj = false;
    bool isFreeRow = false;

    HighsInt start = 0;
    HighsInt end = 0;

    HMpsFF::Parsekey key = checkFirstWord(strline, start, end, word);

    // start of new section?
    if (key != HMpsFF::Parsekey::kNone) {
      highsLogDev(log_options, HighsLogType::kInfo,
                  "readMPS: Read ROWS    OK\n");
      if (!hasobj) {
        highsLogUser(log_options, HighsLogType::kWarning,
                     "No objective row found\n");
        rowname2idx.emplace("artificial_empty_objective", -1);
      };
      return key;
    }

    if (strline[start] == 'G') {
      row_lower.push_back(0.0);
      row_upper.push_back(kHighsInf);
      row_type.push_back(Boundtype::kGe);
    } else if (strline[start] == 'E') {
      row_lower.push_back(0.0);
      row_upper.push_back(0.0);
      row_type.push_back(Boundtype::kEq);
    } else if (strline[start] == 'L') {
      row_lower.push_back(-kHighsInf);
      row_upper.push_back(0.0);
      row_type.push_back(Boundtype::kLe);
    } else if (strline[start] == 'N') {
      if (!hasobj) {
        isobj = true;
        hasobj = true;
        cost_row_location = num_row;
      } else {
        isFreeRow = true;
      }
    } else {
      highsLogUser(log_options, HighsLogType::kError,
                   "Entry in ROWS section of MPS file is of type \"%s\"\n",
                   strline[start]);
      return HMpsFF::Parsekey::kFail;
    }

    std::string rowname = first_word(strline, start + 1);
    HighsInt rowname_end = first_word_end(strline, start + 1);

    // Detect if file is in fixed format.
    if (!is_end(strline, rowname_end)) {
      std::string name = strline.substr(start + 1);
      name = trim(name);
      if (name.size() > 8)
        return HMpsFF::Parsekey::kFail;
      else
        return HMpsFF::Parsekey::kFixedFormat;
    }

    // Do not add to matrix if row is free.
    if (isFreeRow) {
      rowname2idx.emplace(rowname, -2);
      continue;
    }

    // so in rowname2idx -1 is the objective, -2 is all the free rows
    auto ret = rowname2idx.emplace(rowname, isobj ? (-1) : (num_row++));
    // ret is a pair consisting of an iterator to the inserted
    // element (or the already-existing element if no insertion
    // happened) and a bool denoting whether the insertion took place

    // Else is enough here because all free rows are ignored.
    if (!isobj)
      row_names.push_back(rowname);
    else
      objective_name = rowname;

    if (!ret.second) {
      // Duplicate row name
      if (!has_duplicate_row_name_) {
        // This is the first so record it
        has_duplicate_row_name_ = true;
        auto mit = rowname2idx.find(rowname);
        assert(mit != rowname2idx.end());
        duplicate_row_name_ = rowname;
        duplicate_row_name_index0_ = mit->second;
        duplicate_row_name_index1_ = num_row - 1;
      }
    }
  }

  // Hard to imagine how the following lines are executed
  highsLogUser(log_options, HighsLogType::kError,
               "Anomalous exit when parsing BOUNDS section of MPS file\n");
  assert(1 == 0);
  // Update num_row in case there is free rows. They won't be added to the
  // constraint matrix.
  num_row = row_lower.size();
  return HMpsFF::Parsekey::kFail;
}

typename HMpsFF::Parsekey HMpsFF::parseCols(const HighsLogOptions& log_options,
                                            std::istream& file) {
  std::string colname = "";
  std::string strline, word;
  HighsInt rowidx, start, end;
  bool integral_cols = false;
  assert(num_col == 0);
  // Define the scattered value vector, index vector and count
  std::vector<double> col_value;
  std::vector<HighsInt> col_index;
  HighsInt col_count = 0;
  double col_cost = 0;
  col_value.assign(num_row, 0);
  col_index.resize(num_row);

  auto parseName = [&rowidx, this](std::string name) {
    auto mit = rowname2idx.find(name);

    assert(mit != rowname2idx.end());
    rowidx = mit->second;

    if (rowidx >= 0)
      this->num_nz++;
    else
      assert(-1 == rowidx || -2 == rowidx);
  };

  while (getline(file, strline)) {
    double current = getWallTime();
    if (time_limit > 0 && current - start_time > time_limit)
      return HMpsFF::Parsekey::kTimeout;

    if (kAnyFirstNonBlankAsStarImpliesComment) {
      trim(strline);
      if (strline.size() == 0 || strline[0] == '*') continue;
    } else {
      if (strline.size() > 0) {
        // Just look for comment character in column 1
        if (strline[0] == '*') continue;
      }
      trim(strline);
      if (strline.size() == 0) continue;
    }

    HMpsFF::Parsekey key = checkFirstWord(strline, start, end, word);

    // start of new section?
    if (key != Parsekey::kNone) {
      if (num_col) {
        if (col_cost) {
          coeffobj.push_back(std::make_pair(num_col - 1, col_cost));
          col_cost = 0;
        }
        for (HighsInt iEl = 0; iEl < col_count; iEl++) {
          const HighsInt iRow = col_index[iEl];
          assert(col_value[iRow]);
          entries.push_back(
              std::make_tuple(num_col - 1, iRow, col_value[iRow]));
          col_value[iRow] = 0;
        }
        col_count = 0;
      }

      highsLogDev(log_options, HighsLogType::kInfo,
                  "readMPS: Read COLUMNS OK\n");
      return key;
    }

    // check for integrality marker
    std::string marker = first_word(strline, end);
    HighsInt end_marker = first_word_end(strline, end);

    if (marker == "'MARKER'") {
      marker = first_word(strline, end_marker);

      if ((integral_cols && marker != "'INTEND'") ||
          (!integral_cols && marker != "'INTORG'")) {
        highsLogUser(
            log_options, HighsLogType::kError,
            "Integrality marker error in COLUMNS section of MPS file\n");
        return Parsekey::kFail;
      }
      integral_cols = !integral_cols;

      continue;
    }
    // Detect whether the file is in fixed format with spaces in
    // names, even if there are no known examples!
    //
    // end_marker should be the end index of the row name:
    //
    // If the names are at least 8 characters, end_marker should be
    // more than 13 minus the 4 whitespaces we have trimmed from the
    // start so more than 9
    //
    // However, free format MPS can have names with only one character
    // (pyomo.mps). Have to distinguish this from 8-character names
    // with spaces. Best bet is to see whether "marker" is in the set
    // of row names. If it is, then assume that the names are short
    if (end_marker < 9) {
      auto mit = rowname2idx.find(marker);
      if (mit == rowname2idx.end()) {
        // marker is not a row name, so continue to look at name
        std::string name = strline.substr(0, 10);
        // Delete trailing spaces
        name = trim(name);
        if (name.size() > 8) {
          highsLogUser(log_options, HighsLogType::kError,
                       "Row name \"%s\" with spaces exceeds fixed format name "
                       "length of 8\n",
                       name.c_str());
          return HMpsFF::Parsekey::kFail;
        } else {
          highsLogUser(log_options, HighsLogType::kWarning,
                       "Row name \"%s\" with spaces has length %d, so assume "
                       "fixed format\n",
                       name.c_str(), (int)name.size());
          return HMpsFF::Parsekey::kFixedFormat;
        }
      }
    }

    // Test for new column
    if (!(word == colname)) {
      // Record the nonzeros in any previous column
      if (num_col) {
        if (col_cost) {
          coeffobj.push_back(std::make_pair(num_col - 1, col_cost));
          col_cost = 0;
        }
        for (HighsInt iEl = 0; iEl < col_count; iEl++) {
          const HighsInt iRow = col_index[iEl];
          assert(col_value[iRow]);
          entries.push_back(
              std::make_tuple(num_col - 1, iRow, col_value[iRow]));
          col_value[iRow] = 0;
        }
        col_count = 0;
      }
      assert(!col_cost);
      colname = word;
      auto ret = colname2idx.emplace(colname, num_col++);
      col_names.push_back(colname);
      if (!ret.second) {
        // Duplicate col name
        if (!has_duplicate_col_name_) {
          // This is the first so record it
          has_duplicate_col_name_ = true;
          auto mit = colname2idx.find(colname);
          assert(mit != colname2idx.end());
          duplicate_col_name_ = colname;
          duplicate_col_name_index0_ = mit->second;
          duplicate_col_name_index1_ = num_col - 1;
        }
      }

      // Mark the column as integer, according to whether
      // the integral_cols flag is set
      col_integrality.push_back(integral_cols ? HighsVarType::kInteger
                                              : HighsVarType::kContinuous);
      // Mark the column as binary as well
      col_binary.push_back(integral_cols && kintegerVarsInColumnsAreBinary);

      // initialize with default bounds
      col_lower.push_back(0.0);
      col_upper.push_back(kHighsInf);
    }

    assert(num_col > 0);

    // here marker is the row name and end marks its end
    word = "";
    word = first_word(strline, end_marker);
    end = first_word_end(strline, end_marker);

    if (word == "") {
      highsLogUser(log_options, HighsLogType::kError,
                   "No coefficient given for column \"%s\"\n", marker.c_str());
      return HMpsFF::Parsekey::kFail;
    }

    auto mit = rowname2idx.find(marker);
    if (mit == rowname2idx.end()) {
      highsLogUser(
          log_options, HighsLogType::kWarning,
          "Row name \"%s\" in COLUMNS section is not defined: ignored\n",
          marker.c_str());
    } else {
      double value = atof(word.c_str());
      if (value) {
        parseName(marker);  // rowidx set and num_nz incremented
        if (rowidx >= 0) {
          if (col_value[rowidx]) {
            // Ignore duplicate entry
            num_nz--;
            highsLogUser(log_options, HighsLogType::kWarning,
                         "Column \"%s\" has duplicate nonzero in row \"%s\"\n",
                         colname.c_str(), marker.c_str());
          } else {
            col_value[rowidx] = value;
            col_index[col_count++] = rowidx;
          }
        } else if (rowidx == -1) {
          // Ignore duplicate entry
          if (col_cost) {
            highsLogUser(log_options, HighsLogType::kWarning,
                         "Column \"%s\" has duplicate nonzero in row \"%s\"\n",
                         colname.c_str(), objective_name.c_str());
          } else {
            col_cost = value;
          }
        }
      }
    }

    if (!is_end(strline, end)) {
      // parse second coefficient
      marker = first_word(strline, end);
      if (word == "") {
        highsLogUser(log_options, HighsLogType::kError,
                     "No coefficient given for column \"%s\"\n",
                     marker.c_str());
        return HMpsFF::Parsekey::kFail;
      }
      end_marker = first_word_end(strline, end);

      // here marker is the row name and end marks its end
      word = "";
      end_marker++;
      word = first_word(strline, end_marker);
      end = first_word_end(strline, end_marker);

      assert(is_end(strline, end));

      auto mit = rowname2idx.find(marker);
      if (mit == rowname2idx.end()) {
        highsLogUser(
            log_options, HighsLogType::kWarning,
            "Row name \"%s\" in COLUMNS section is not defined: ignored\n",
            marker.c_str());
        continue;
      };
      double value = atof(word.c_str());
      if (value) {
        parseName(marker);  // rowidx set and num_nz incremented
        if (rowidx >= 0) {
          if (col_value[rowidx]) {
            // Ignore duplicate entry
            num_nz--;
            highsLogUser(log_options, HighsLogType::kWarning,
                         "Column \"%s\" has duplicate nonzero in row \"%s\"\n",
                         colname.c_str(), marker.c_str());
          } else {
            col_value[rowidx] = value;
            col_index[col_count++] = rowidx;
          }
        } else if (rowidx == -1) {
          // Ignore duplicate entry
          if (col_cost) {
            highsLogUser(log_options, HighsLogType::kWarning,
                         "Column \"%s\" has duplicate nonzero in row \"%s\"\n",
                         colname.c_str(), objective_name.c_str());
          } else {
            col_cost = value;
          }
        }
      }
    }
  }

  return Parsekey::kFail;
}

HMpsFF::Parsekey HMpsFF::parseRhs(const HighsLogOptions& log_options,
                                  std::istream& file) {
  std::string strline;

  auto parseName = [this](const std::string& name, HighsInt& rowidx,
                          bool& has_entry) {
    auto mit = rowname2idx.find(name);

    assert(mit != rowname2idx.end());
    rowidx = mit->second;

    assert(rowidx < num_row);

    if (rowidx > -1) {
      has_entry = has_row_entry_[rowidx];
    } else {
      assert(rowidx == -1);
      has_entry = has_obj_entry_;
    }
  };

  auto addRhs = [this](double val, HighsInt rowidx) {
    if (rowidx > -1) {
      if (row_type[rowidx] == Boundtype::kEq ||
          row_type[rowidx] == Boundtype::kLe) {
        assert(size_t(rowidx) < row_upper.size());
        row_upper[rowidx] = val;
      }
      if (row_type[rowidx] == Boundtype::kEq ||
          row_type[rowidx] == Boundtype::kGe) {
        assert(size_t(rowidx) < row_lower.size());
        row_lower[rowidx] = val;
      }
      has_row_entry_[rowidx] = true;
    } else {
      // objective shift
      assert(rowidx == -1);
      obj_offset = -val;
      has_obj_entry_ = true;
    }
  };

  // Initialise tracking for duplicate entries
  has_row_entry_.assign(num_row, false);
  has_obj_entry_ = false;
  bool has_entry = false;

  while (getline(file, strline)) {
    double current = getWallTime();
    if (time_limit > 0 && current - start_time > time_limit)
      return HMpsFF::Parsekey::kTimeout;

    if (kAnyFirstNonBlankAsStarImpliesComment) {
      trim(strline);
      if (strline.size() == 0 || strline[0] == '*') continue;
    } else {
      if (strline.size() > 0) {
        // Just look for comment character in column 1
        if (strline[0] == '*') continue;
      }
      trim(strline);
      if (strline.size() == 0) continue;
    }

    HighsInt begin = 0;
    HighsInt end = 0;
    std::string word;
    HMpsFF::Parsekey key = checkFirstWord(strline, begin, end, word);

    // start of new section?
    if (key != Parsekey::kNone && key != Parsekey::kRhs) {
      highsLogDev(log_options, HighsLogType::kInfo,
                  "readMPS: Read RHS     OK\n");
      return key;
    }

    // Ignore lack of name for SIF format;
    // we know we have this case when "word" is a row name
    if ((key == Parsekey::kNone) && (key != Parsekey::kRhs) &&
        (rowname2idx.find(word) != rowname2idx.end())) {
      end = begin;
    }

    HighsInt rowidx;

    std::string marker = first_word(strline, end);
    HighsInt end_marker = first_word_end(strline, end);

    // here marker is the row name and end marks its end
    word = "";
    word = first_word(strline, end_marker);
    end = first_word_end(strline, end_marker);

    if (word == "") {
      highsLogUser(log_options, HighsLogType::kError,
                   "No bound given for row \"%s\"\n", marker.c_str());
      return HMpsFF::Parsekey::kFail;
    }

    auto mit = rowname2idx.find(marker);

    // SIF format sometimes has the name of the MPS file
    // prepended to the RHS entry; remove it here if
    // that's the case. "word" will then hold the marker,
    // so also get new "word" and "end" values
    if (mit == rowname2idx.end()) {
      if (marker == mps_name) {
        marker = word;
        end_marker = end;
        word = "";
        word = first_word(strline, end_marker);
        end = first_word_end(strline, end_marker);
        if (word == "") {
          highsLogUser(log_options, HighsLogType::kError,
                       "No bound given for SIF row \"%s\"\n", marker.c_str());
          return HMpsFF::Parsekey::kFail;
        }
        mit = rowname2idx.find(marker);
      }
    }

    if (mit == rowname2idx.end()) {
      highsLogUser(log_options, HighsLogType::kWarning,
                   "Row name \"%s\" in RHS section is not defined: ignored\n",
                   marker.c_str());
    } else {
      parseName(marker, rowidx, has_entry);
      if (has_entry) {
        highsLogUser(log_options, HighsLogType::kWarning,
                     "Row name \"%s\" in RHS section has duplicate definition: "
                     "ignored\n",
                     marker.c_str());
      } else {
        double value = atof(word.c_str());
        addRhs(value, rowidx);
      }
    }

    if (!is_end(strline, end)) {
      // parse second coefficient
      marker = first_word(strline, end);
      if (word == "") {
        highsLogUser(log_options, HighsLogType::kError,
                     "No coefficient given for rhs of row \"%s\"\n",
                     marker.c_str());
        return HMpsFF::Parsekey::kFail;
      }
      end_marker = first_word_end(strline, end);

      // here marker is the row name and end marks its end
      word = "";
      end_marker++;
      word = first_word(strline, end_marker);
      end = first_word_end(strline, end_marker);

      assert(is_end(strline, end));

      auto mit = rowname2idx.find(marker);
      if (mit == rowname2idx.end()) {
        highsLogUser(log_options, HighsLogType::kWarning,
                     "Row name \"%s\" in RHS section is not defined: ignored\n",
                     marker.c_str());
        continue;
      };

      parseName(marker, rowidx, has_entry);
      if (has_entry) {
        highsLogUser(log_options, HighsLogType::kWarning,
                     "Row name \"%s\" in RHS section has duplicate definition: "
                     "ignored\n",
                     marker.c_str());
      } else {
        double value = atof(word.c_str());
        addRhs(value, rowidx);
      }
    }
  }

  return Parsekey::kFail;
}

HMpsFF::Parsekey HMpsFF::parseBounds(const HighsLogOptions& log_options,
                                     std::istream& file) {
  std::string strline, word;

  HighsInt num_mi = 0;
  HighsInt num_pl = 0;
  HighsInt num_bv = 0;
  HighsInt num_li = 0;
  HighsInt num_ui = 0;
  HighsInt num_si = 0;
  HighsInt num_sc = 0;

  std::vector<bool> has_lower;
  std::vector<bool> has_upper;
  has_lower.assign(num_col, false);
  has_upper.assign(num_col, false);

  while (getline(file, strline)) {
    double current = getWallTime();
    if (time_limit > 0 && current - start_time > time_limit)
      return HMpsFF::Parsekey::kTimeout;

    if (kAnyFirstNonBlankAsStarImpliesComment) {
      trim(strline);
      if (strline.size() == 0 || strline[0] == '*') continue;
    } else {
      if (strline.size() > 0) {
        // Just look for comment character in column 1
        if (strline[0] == '*') continue;
      }
      trim(strline);
      if (strline.size() == 0) continue;
    }

    HighsInt begin = 0;
    HighsInt end = 0;
    std::string word;
    HMpsFF::Parsekey key = checkFirstWord(strline, begin, end, word);

    // start of new section?
    if (key != Parsekey::kNone) {
      if (num_mi)
        highsLogUser(
            log_options, HighsLogType::kInfo,
            "Number of MI entries in BOUNDS section is %" HIGHSINT_FORMAT "\n",
            num_mi);
      if (num_pl)
        highsLogUser(
            log_options, HighsLogType::kInfo,
            "Number of PL entries in BOUNDS section is %" HIGHSINT_FORMAT "\n",
            num_pl);
      if (num_bv)
        highsLogUser(
            log_options, HighsLogType::kInfo,
            "Number of BV entries in BOUNDS section is %" HIGHSINT_FORMAT "\n",
            num_bv);
      if (num_li)
        highsLogUser(
            log_options, HighsLogType::kInfo,
            "Number of LI entries in BOUNDS section is %" HIGHSINT_FORMAT "\n",
            num_li);
      if (num_ui)
        highsLogUser(
            log_options, HighsLogType::kInfo,
            "Number of UI entries in BOUNDS section is %" HIGHSINT_FORMAT "\n",
            num_ui);
      if (num_si)
        highsLogUser(
            log_options, HighsLogType::kInfo,
            "Number of SI entries in BOUNDS section is %" HIGHSINT_FORMAT "\n",
            num_si);
      if (num_sc)
        highsLogUser(
            log_options, HighsLogType::kInfo,
            "Number of SC entries in BOUNDS section is %" HIGHSINT_FORMAT "\n",
            num_sc);
      highsLogDev(log_options, HighsLogType::kInfo,
                  "readMPS: Read BOUNDS  OK\n");
      return key;
    }
    bool is_lb = false;
    bool is_ub = false;
    bool is_integral = false;
    bool is_semi = false;
    bool is_defaultbound = false;
    if (word == "UP")  // lower bound
      is_ub = true;
    else if (word == "LO")  // upper bound
      is_lb = true;
    else if (word == "FX")  // fixed
    {
      is_lb = true;
      is_ub = true;
    } else if (word == "MI")  // infinite lower bound
    {
      is_lb = true;
      is_defaultbound = true;
      num_mi++;
    } else if (word == "PL")  // infinite upper bound (redundant)
    {
      is_ub = true;
      is_defaultbound = true;
      num_pl++;
    } else if (word == "BV")  // binary
    {
      is_lb = true;
      is_ub = true;
      is_integral = true;
      is_defaultbound = true;
      num_bv++;
    } else if (word == "LI")  // integer lower bound
    {
      is_lb = true;
      is_integral = true;
      num_li++;
    } else if (word == "UI")  // integer upper bound
    {
      is_ub = true;
      is_integral = true;
      num_ui++;
    } else if (word == "FR")  // free variable
    {
      is_lb = true;
      is_ub = true;
      is_defaultbound = true;
    } else if (word == "SI")  // semi-integer variable
    {
      is_ub = true;
      is_integral = true;
      is_semi = true;
      num_si++;
    } else if (word == "SC")  // semi-continuous variable
    {
      is_ub = true;
      is_semi = true;
      num_sc++;
    } else {
      highsLogUser(log_options, HighsLogType::kError,
                   "Entry in BOUNDS section of MPS file is of type \"%s\"\n",
                   word.c_str());
      return HMpsFF::Parsekey::kFail;
    }

    std::string bound_name = first_word(strline, end);
    HighsInt end_bound_name = first_word_end(strline, end);

    std::string marker;
    HighsInt end_marker;
    if (colname2idx.find(bound_name) != colname2idx.end()) {
      // SIF format might not have the bound name, so skip
      // it here if we found the marker instead
      marker = bound_name;
      end_marker = end_bound_name;
    } else {
      // The first word is the bound name, which should be ignored.
      marker = first_word(strline, end_bound_name);
      end_marker = first_word_end(strline, end_bound_name);
    }

    // BOUNDS: get column index from name, without adding new column
    // if not existing yet
    HighsInt colidx = getColIdx(marker, false);
    if (colidx < 0) {
      // add new column if did not exist yet
      colidx = getColIdx(marker, true);
      assert(colidx == num_col - 1);
      has_lower.push_back(false);
      has_upper.push_back(false);
    }

    // Determine whether this entry yields a duplicate bound
    // definition
    if ((is_lb && has_lower[colidx]) || (is_ub && has_upper[colidx])) {
      highsLogUser(log_options, HighsLogType::kWarning,
                   "Column name \"%s\" in BOUNDS section has duplicate "
                   "definition: ignored\n",
                   marker.c_str());
      continue;
    }

    if (is_defaultbound) {
      // MI, PL, BV or FR
      if (is_integral)
      // binary: BV
      {
        if (!is_lb || !is_ub) {
          highsLogUser(log_options, HighsLogType::kError,
                       "BV row %s but [is_lb, is_ub] = [%1" HIGHSINT_FORMAT
                       ", %1" HIGHSINT_FORMAT "]\n",
                       marker.c_str(), is_lb, is_ub);
          assert(is_lb && is_ub);
          return HMpsFF::Parsekey::kFail;
        }
        assert(is_lb && is_ub);
        // Mark the column as integer and binary
        col_integrality[colidx] = HighsVarType::kInteger;
        col_binary[colidx] = true;
        assert(col_lower[colidx] == 0.0);
        col_upper[colidx] = 1.0;
      } else {
        // continuous: MI, PL or FR
        col_binary[colidx] = false;
        if (is_lb) col_lower[colidx] = -kHighsInf;
        if (is_ub) col_upper[colidx] = kHighsInf;
      }
      if (is_lb) has_lower[colidx] = true;
      if (is_ub) has_upper[colidx] = true;
      continue;
    }
    // Bounds now are UP, LO, FX, LI, UI, SI or SC
    // here marker is the col name and end marks its end
    word = "";
    word = first_word(strline, end_marker);
    end = first_word_end(strline, end_marker);

    if (word == "") {
      highsLogUser(log_options, HighsLogType::kError,
                   "No bound given for row \"%s\"\n", marker.c_str());
      return HMpsFF::Parsekey::kFail;
    }
    double value = atof(word.c_str());
    if (is_integral) {
      assert(is_lb || is_ub || is_semi);
      // Must be LI or UI, and value should be integer
      HighsInt i_value = static_cast<HighsInt>(value);
      double dl = value - i_value;
      if (dl)
        highsLogUser(log_options, HighsLogType::kError,
                     "Bound for LI/UI/SI column \"%s\" is %g: not integer\n",
                     marker.c_str(), value);
      if (is_semi) {
        // Bound marker SI defines the column as semi-integer
        col_integrality[colidx] = HighsVarType::kSemiInteger;
      } else {
        // Bound marker LI or UI defines the column as integer
        col_integrality[colidx] = HighsVarType::kInteger;
      }
    } else if (is_semi) {
      // Bound marker SC defines the column as semi-continuous
      col_integrality[colidx] = HighsVarType::kSemiContinuous;
    }
    // Assign the bounds that have been read
    if (is_lb) {
      col_lower[colidx] = value;
      has_lower[colidx] = true;
    }
    if (is_ub) {
      col_upper[colidx] = value;
      has_upper[colidx] = true;
    }
    // Column is not binary by default
    col_binary[colidx] = false;
  }
  return Parsekey::kFail;
}

HMpsFF::Parsekey HMpsFF::parseRanges(const HighsLogOptions& log_options,
                                     std::istream& file) {
  std::string strline, word;

  auto parseName = [this](const std::string& name, HighsInt& rowidx) {
    auto mit = rowname2idx.find(name);

    assert(mit != rowname2idx.end());
    rowidx = mit->second;

    assert(rowidx < num_row);
  };

  auto addRhs = [this](double val, HighsInt& rowidx) {
    if ((row_type[rowidx] == Boundtype::kEq && val < 0) ||
        row_type[rowidx] == Boundtype::kLe) {
      assert(row_upper.at(rowidx) < kHighsInf);
      row_lower.at(rowidx) = row_upper.at(rowidx) - fabs(val);
    } else if ((row_type[rowidx] == Boundtype::kEq && val > 0) ||
               row_type[rowidx] == Boundtype::kGe) {
      assert(row_lower.at(rowidx) > (-kHighsInf));
      row_upper.at(rowidx) = row_lower.at(rowidx) + fabs(val);
    }
    has_row_entry_[rowidx] = true;
  };

  // Initialise tracking for duplicate entries
  has_row_entry_.assign(num_row, false);

  while (getline(file, strline)) {
    double current = getWallTime();
    if (time_limit > 0 && current - start_time > time_limit)
      return HMpsFF::Parsekey::kTimeout;

    if (kAnyFirstNonBlankAsStarImpliesComment) {
      trim(strline);
      if (strline.size() == 0 || strline[0] == '*') continue;
    } else {
      if (strline.size() > 0) {
        // Just look for comment character in column 1
        if (strline[0] == '*') continue;
      }
      trim(strline);
      if (strline.size() == 0) continue;
    }

    HighsInt begin, end;
    std::string word;
    HMpsFF::Parsekey key = checkFirstWord(strline, begin, end, word);

    if (key != Parsekey::kNone) {
      highsLogDev(log_options, HighsLogType::kInfo,
                  "readMPS: Read RANGES  OK\n");
      return key;
    }

    HighsInt rowidx;

    std::string marker = first_word(strline, end);
    HighsInt end_marker = first_word_end(strline, end);

    // here marker is the row name and end marks its end
    word = "";
    word = first_word(strline, end_marker);
    end = first_word_end(strline, end_marker);

    if (word == "") {
      highsLogUser(log_options, HighsLogType::kError,
                   "No range given for row \"%s\"\n", marker.c_str());
      return HMpsFF::Parsekey::kFail;
    }

    auto mit = rowname2idx.find(marker);
    if (mit == rowname2idx.end()) {
      highsLogUser(
          log_options, HighsLogType::kWarning,
          "Row name \"%s\" in RANGES section is not defined: ignored\n",
          marker.c_str());
    } else {
      parseName(marker, rowidx);
      if (rowidx < 0) {
        highsLogUser(
            log_options, HighsLogType::kWarning,
            "Row name \"%s\" in RANGES section is not valid: ignored\n",
            marker.c_str());
      } else if (has_row_entry_[rowidx]) {
        highsLogUser(log_options, HighsLogType::kWarning,
                     "Row name \"%s\" in RANGES section has duplicate "
                     "definition: ignored\n",
                     marker.c_str());
      } else {
        double value = atof(word.c_str());
        addRhs(value, rowidx);
      }
    }

    if (!is_end(strline, end)) {
      std::string marker = first_word(strline, end);
      HighsInt end_marker = first_word_end(strline, end);

      // here marker is the row name and end marks its end
      word = "";
      word = first_word(strline, end_marker);
      end = first_word_end(strline, end_marker);

      if (word == "") {
        highsLogUser(log_options, HighsLogType::kError,
                     "No range given for row \"%s\"\n", marker.c_str());
        return HMpsFF::Parsekey::kFail;
      }

      auto mit = rowname2idx.find(marker);
      if (mit == rowname2idx.end()) {
        highsLogUser(
            log_options, HighsLogType::kWarning,
            "Row name \"%s\" in RANGES section is not defined: ignored\n",
            marker.c_str());
      } else {
        parseName(marker, rowidx);
        if (rowidx < 0) {
          highsLogUser(
              log_options, HighsLogType::kWarning,
              "Row name \"%s\" in RANGES section is not valid: ignored\n",
              marker.c_str());
        } else if (has_row_entry_[rowidx]) {
          highsLogUser(log_options, HighsLogType::kWarning,
                       "Row name \"%s\" in RANGES section has duplicate "
                       "definition: ignored\n",
                       marker.c_str());
        } else {
          double value = atof(word.c_str());
          addRhs(value, rowidx);
        }
      }

      if (!is_end(strline, end)) {
        highsLogUser(log_options, HighsLogType::kError,
                     "Unknown specifiers in RANGES section for row \"%s\"\n",
                     marker.c_str());
        return HMpsFF::Parsekey::kFail;
      }
    }
  }

  return HMpsFF::Parsekey::kFail;
}

typename HMpsFF::Parsekey HMpsFF::parseHessian(
    const HighsLogOptions& log_options, std::istream& file,
    const HMpsFF::Parsekey keyword) {
  // Parse Hessian information from QUADOBJ or QMATRIX
  // section according to keyword
  const bool qmatrix = keyword == HMpsFF::Parsekey::kQmatrix;
  std::string section_name;
  if (qmatrix) {
    section_name = "QMATRIX";
  } else if (keyword == HMpsFF::Parsekey::kQuadobj) {
    section_name = "QUADOBJ";
  }
  std::string strline;
  std::string col_name;
  std::string row_name;
  std::string coeff_name;
  HighsInt end_row_name;
  HighsInt end_coeff_name;
  HighsInt colidx, rowidx;

  while (getline(file, strline)) {
    double current = getWallTime();
    if (time_limit > 0 && current - start_time > time_limit)
      return HMpsFF::Parsekey::kTimeout;
    if (kAnyFirstNonBlankAsStarImpliesComment) {
      trim(strline);
      if (strline.size() == 0 || strline[0] == '*') continue;
    } else {
      if (strline.size() > 0) {
        // Just look for comment character in column 1
        if (strline[0] == '*') continue;
      }
      trim(strline);
      if (strline.size() == 0) continue;
    }

    HighsInt begin = 0;
    HighsInt end = 0;
    HMpsFF::Parsekey key = checkFirstWord(strline, begin, end, col_name);

    // start of new section?
    if (key != Parsekey::kNone) {
      highsLogDev(log_options, HighsLogType::kInfo, "readMPS: Read %s OK\n",
                  section_name.c_str());
      return key;
    }

    // Get the column index from the name
    colidx = getColIdx(col_name);
    assert(colidx >= 0 && colidx < num_col);

    // Loop over the maximum of two entries per row of the file
    for (int entry = 0; entry < 2; entry++) {
      // Get the row name
      row_name = "";
      row_name = first_word(strline, end);
      end_row_name = first_word_end(strline, end);

      if (row_name == "") break;

      coeff_name = "";
      coeff_name = first_word(strline, end_row_name);
      end_coeff_name = first_word_end(strline, end_row_name);

      if (coeff_name == "") {
        highsLogUser(
            log_options, HighsLogType::kError,
            "%s has no coefficient for entry \"%s\" in column \"%s\"\n",
            section_name.c_str(), row_name.c_str(), col_name.c_str());
        return HMpsFF::Parsekey::kFail;
      }

      rowidx = getColIdx(row_name);
      assert(rowidx >= 0 && rowidx < num_col);

      double coeff = atof(coeff_name.c_str());
      if (coeff) {
        if (qmatrix) {
          // QMATRIX has the whole Hessian, so store the entry if the
          // entry is in the lower triangle
          if (rowidx >= colidx)
            q_entries.push_back(std::make_tuple(rowidx, colidx, coeff));
        } else {
          // QSECTION and QUADOBJ has the lower triangle of the
          // Hessian
          q_entries.push_back(std::make_tuple(rowidx, colidx, coeff));
          //          if (rowidx != colidx)
          //            q_entries.push_back(std::make_tuple(colidx, rowidx,
          //            coeff));
        }
      }
      end = end_coeff_name;
      // Don't read more if end of line reached
      if (end == (HighsInt)strline.length()) break;
    }
  }

  return HMpsFF::Parsekey::kFail;
}

typename HMpsFF::Parsekey HMpsFF::parseQuadRows(
    const HighsLogOptions& log_options, std::istream& file,
    const HMpsFF::Parsekey keyword) {
  // Parse Hessian information from QSECTION or QCMATRIX
  // section according to keyword
  const bool qcmatrix = keyword == HMpsFF::Parsekey::kQcmatrix;
  std::string section_name;
  if (qcmatrix) {
    section_name = "QCMATRIX";
  } else {
    section_name = "QSECTION";
  }
  std::string strline;
  std::string col_name;
  std::string row_name;
  std::string coeff_name;
  HighsInt end_row_name;
  HighsInt end_coeff_name;
  HighsInt rowidx;            // index of quadratic row
  HighsInt qcolidx, qrowidx;  // indices in quadratic coefs matrix

  // Get row name from section argument
  std::string rowname = first_word(section_args, 0);
  if (rowname.empty()) {
    highsLogUser(log_options, HighsLogType::kError,
                 "No row name given in argument of %s\n", section_name.c_str());
    return HMpsFF::Parsekey::kFail;
  }

  auto mit = rowname2idx.find(rowname);
  // if row of section does not exist or is free (index -2), then skip
  if (mit == rowname2idx.end() || mit->second == -2) {
    if (mit == rowname2idx.end())
      highsLogUser(log_options, HighsLogType::kWarning,
                   "Row name \"%s\" in %s section is not defined: ignored\n",
                   rowname.c_str(), section_name.c_str());
    // read lines until start of new section
    while (getline(file, strline)) {
      HighsInt begin = 0;
      HighsInt end = 0;
      HMpsFF::Parsekey key = checkFirstWord(strline, begin, end, col_name);

      // start of new section?
      if (key != Parsekey::kNone) {
        highsLogDev(log_options, HighsLogType::kInfo, "readMPS: Read %s  OK\n",
                    section_name.c_str());
        return key;
      }
    }
    return Parsekey::kFail;  // unexpected end of file
  }
  rowidx = mit->second;
  assert(rowidx >= -1);
  assert(rowidx < num_row);

  if (rowidx >= 0) qrows_entries.resize(num_row);
  assert(rowidx == -1 || (HighsInt)qrows_entries.size() == num_row);

  auto& qentries = (rowidx == -1 ? q_entries : qrows_entries[rowidx]);

  while (getline(file, strline)) {
    double current = getWallTime();
    if (time_limit > 0 && current - start_time > time_limit)
      return HMpsFF::Parsekey::kTimeout;
    if (kAnyFirstNonBlankAsStarImpliesComment) {
      trim(strline);
      if (strline.size() == 0 || strline[0] == '*') continue;
    } else {
      if (strline.size() > 0) {
        // Just look for comment character in column 1
        if (strline[0] == '*') continue;
      }
      trim(strline);
      if (strline.size() == 0) continue;
    }

    HighsInt begin = 0;
    HighsInt end = 0;
    HMpsFF::Parsekey key = checkFirstWord(strline, begin, end, col_name);

    // start of new section?
    if (key != Parsekey::kNone) {
      highsLogDev(log_options, HighsLogType::kInfo, "readMPS: Read %s  OK\n",
                  section_name.c_str());
      return key;
    }

    // Get the column index
    qcolidx = getColIdx(col_name);
    assert(qcolidx >= 0 && qcolidx < num_col);

    // Loop over the maximum of two entries per row of the file
    for (int entry = 0; entry < 2; entry++) {
      // Get the row name
      row_name = "";
      row_name = first_word(strline, end);
      end_row_name = first_word_end(strline, end);

      if (row_name == "") break;

      coeff_name = "";
      coeff_name = first_word(strline, end_row_name);
      end_coeff_name = first_word_end(strline, end_row_name);

      if (coeff_name == "") {
        highsLogUser(
            log_options, HighsLogType::kError,
            "%s has no coefficient for entry \"%s\" in column \"%s\"\n",
            section_name.c_str(), row_name.c_str(), col_name.c_str());
        return HMpsFF::Parsekey::kFail;
      }

      qrowidx = getColIdx(row_name);
      assert(qrowidx >= 0 && qrowidx < num_col);

      double coeff = atof(coeff_name.c_str());
      if (coeff) {
        if (qcmatrix) {
          // QCMATRIX has the whole Hessian, so store the entry if the
          // entry is in the lower triangle
          if (qrowidx >= qcolidx)
            qentries.push_back(std::make_tuple(qrowidx, qcolidx, coeff));
        } else {
          // QSECTION has the lower triangle of the Hessian
          qentries.push_back(std::make_tuple(qrowidx, qcolidx, coeff));
        }
      }
      end = end_coeff_name;
      // Don't read more if end of line reached
      if (end == (HighsInt)strline.length()) break;
    }
  }

  return HMpsFF::Parsekey::kFail;
}

typename HMpsFF::Parsekey HMpsFF::parseCones(const HighsLogOptions& log_options,
                                             std::istream& file) {
  HighsInt end = 0;

  // first argument should be cone name
  std::string conename = first_word(section_args, end);
  end = first_word_end(section_args, end);

  if (conename.empty()) {
    highsLogUser(log_options, HighsLogType::kError,
                 "Cone name missing in CSECTION\n");
    return HMpsFF::Parsekey::kFail;
  }

  // second argument is cone parameter, but is optional
  // third argument is cone type
  std::string secondarg = first_word(section_args, end);
  end = first_word_end(section_args, end);

  std::string thirdarg = first_word(section_args, end);
  end = first_word_end(section_args, end);

  std::string coneparam = "0.0";
  std::string conetypestr;
  if (thirdarg.empty()) {
    conetypestr = secondarg;
  } else {
    coneparam = secondarg;
    conetypestr = thirdarg;
  }

  if (conetypestr.empty()) {
    highsLogUser(log_options, HighsLogType::kError,
                 "Cone type missing in CSECTION %s\n", section_args.c_str());
    return HMpsFF::Parsekey::kFail;
  }

  ConeType conetype;
  if (conetypestr == "ZERO")
    conetype = ConeType::kZero;
  else if (conetypestr == "QUAD")
    conetype = ConeType::kQuad;
  else if (conetypestr == "RQUAD")
    conetype = ConeType::kRQuad;
  else if (conetypestr == "PEXP")
    conetype = ConeType::kPExp;
  else if (conetypestr == "PPOW")
    conetype = ConeType::kPPow;
  else if (conetypestr == "DEXP")
    conetype = ConeType::kDExp;
  else if (conetypestr == "DPOW")
    conetype = ConeType::kDPow;
  else {
    highsLogUser(log_options, HighsLogType::kError,
                 "Unrecognized cone type %s\n", conetypestr.c_str());
    return HMpsFF::Parsekey::kFail;
  }

  cone_name.push_back(conename);
  cone_type.push_back(conetype);
  cone_param.push_back(atof(coneparam.c_str()));
  cone_entries.push_back(std::vector<HighsInt>());

  // now parse the cone entries: one column per line
  std::string strline;
  while (getline(file, strline)) {
    double current = getWallTime();
    if (time_limit > 0 && current - start_time > time_limit)
      return HMpsFF::Parsekey::kTimeout;

    if (kAnyFirstNonBlankAsStarImpliesComment) {
      trim(strline);
      if (strline.size() == 0 || strline[0] == '*') continue;
    } else {
      if (strline.size() > 0) {
        // Just look for comment character in column 1
        if (strline[0] == '*') continue;
      }
      trim(strline);
      if (strline.size() == 0) continue;
    }

    HighsInt begin;
    std::string colname;
    HMpsFF::Parsekey key = checkFirstWord(strline, begin, end, colname);

    if (key != Parsekey::kNone) {
      highsLogDev(log_options, HighsLogType::kInfo,
                  "readMPS: Read CSECTION OK\n");
      return key;
    }

    // colname -> colidx
    HighsInt colidx = getColIdx(colname);
    assert(colidx >= 0);
    assert(colidx < num_col);

    cone_entries.back().push_back(colidx);
  }

  return HMpsFF::Parsekey::kFail;
}

typename HMpsFF::Parsekey HMpsFF::parseSos(const HighsLogOptions& log_options,
                                           std::istream& file,
                                           const HMpsFF::Parsekey keyword) {
  std::string strline, word;

  while (getline(file, strline)) {
    double current = getWallTime();
    if (time_limit > 0 && current - start_time > time_limit)
      return HMpsFF::Parsekey::kTimeout;

    if (kAnyFirstNonBlankAsStarImpliesComment) {
      trim(strline);
      if (strline.size() == 0 || strline[0] == '*') continue;
    } else {
      if (strline.size() > 0) {
        // Just look for comment character in column 1
        if (strline[0] == '*') continue;
      }
      trim(strline);
      if (strline.size() == 0) continue;
    }

    HighsInt begin, end;
    std::string word;
    HMpsFF::Parsekey key = checkFirstWord(strline, begin, end, word);

    if (key != Parsekey::kNone) {
      highsLogDev(log_options, HighsLogType::kInfo,
                  "readMPS: Read SETS    OK\n");
      return key;
    }

    if (word == "S1" || word == "S2") {
      /* a new SOS is starting */
      std::string sosname = first_word(strline, end);

      if (sosname.empty()) {
        highsLogUser(log_options, HighsLogType::kError,
                     "No name given for SOS\n");
        return HMpsFF::Parsekey::kFail;
      }

      sos_type.push_back(word[1] == '1' ? 1 : 2);
      sos_name.push_back(sosname);
      sos_entries.push_back(std::vector<std::pair<HighsInt, double> >());
      continue;
    }

    /* a SOS is continuing
     * word is currently the column name and there may be a weight following
     */
    if (sos_entries.empty()) {
      highsLogUser(log_options, HighsLogType::kError,
                   "SOS type specification missing before %s.\n",
                   strline.c_str());
      return HMpsFF::Parsekey::kFail;
    }

    std::string colname;

    if (keyword == HMpsFF::Parsekey::kSos) {
      // first word is column index
      colname = word;
    } else {
      // first word is SOS name, second word is colname, third word is weight
      // we expect SOS definitions to be contiguous for now
      if (word != sos_name.back()) {
        highsLogUser(log_options, HighsLogType::kError,
                     "SOS specification for SOS %s mixed with SOS %s. This is "
                     "currently not supported.\n",
                     sos_name.back().c_str(), word.c_str());
        return HMpsFF::Parsekey::kFail;
      }
      if (is_end(strline, end)) {
        highsLogUser(log_options, HighsLogType::kError,
                     "Missing variable in SOS specification line %s.\n",
                     strline.c_str());
        return HMpsFF::Parsekey::kFail;
      }
      colname = first_word(strline, end);
      end = first_word_end(strline, end);
    }

    // colname -> colidx
    HighsInt colidx = getColIdx(colname);
    assert(colidx >= 0);
    assert(colidx < num_col);

    // last word is weight, allow to omit
    double weight = 0.0;
    if (!is_end(strline, end)) {
      word = first_word(strline, end);
      weight = atof(word.c_str());
    }

    sos_entries.back().push_back(std::make_pair(colidx, weight));
  }

  return HMpsFF::Parsekey::kFail;
}

bool HMpsFF::allZeroed(const std::vector<double>& value) {
  for (HighsInt iRow = 0; iRow < num_row; iRow++)
    if (value[iRow]) return false;
  return true;
}

}  // namespace free_format_parser
