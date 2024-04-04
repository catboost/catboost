#ifndef __SRC_LIB_MATRIX_HPP__
#define __SRC_LIB_MATRIX_HPP__

#include <cassert>
#include <vector>

#include "vector.hpp"

#ifdef OPENMP
#include "omp.h"
#endif

struct MatrixBase {
  HighsInt num_row;
  HighsInt num_col;
  std::vector<HighsInt> start;
  std::vector<HighsInt> index;
  std::vector<double> value;

  Vector& mat_vec(const Vector& other, Vector& target) const {
    return mat_vec_seq(other, target);
  }

  Vector& mat_vec_seq(const Vector& other, Vector& target) const {
    target.reset();
    for (HighsInt i = 0; i < other.num_nz; i++) {
      HighsInt col = other.index[i];
      for (HighsInt idx = start[col]; idx < start[col + 1]; idx++) {
        HighsInt row = index[idx];
        target.value[row] += value[idx] * other.value[col];
      }
    }
    target.resparsify();
    return target;
  }

  Vector mat_vec(const Vector& other) {
    Vector result(num_row);
    mat_vec(other, result);
    return result;
  }

  Vector vec_mat(HighsInt* idx, double* val, HighsInt nnz) {
    Vector result(num_col);
    for (HighsInt i = 0; i < num_col; i++) {
      double dot = 0.0;
      // HighsInt idx_other = 0;
      // HighsInt idx_this = start[i];
      // while (idx_this < start[i+1] && idx_other < nnz) {
      //    if (idx[idx_other] == index[idx_this]) {
      //       dot += val[idx_other] * value[idx_this];
      //    } else if (idx[idx_other] < index[idx_this]) {
      //       idx_other++;
      //    } else {
      //       idx_this++;
      //    }
      // }

      for (HighsInt j = start[i]; j < start[i + 1]; j++) {
        // does the vector have an entry for index index[j]?
        double other_value = 0.0;
        for (HighsInt k = 0; k < nnz; k++) {
          if (idx[k] == index[j]) {
            other_value = val[k];
            break;
          }
        }

        dot += other_value * value[j];
      }

      if (dot != 0.0) {
        result.value[i] = dot;
        result.index[result.num_nz] = i;
        result.num_nz++;
      }
    }
    return result;
  }

  Vector& vec_mat(const Vector& other, Vector& target) const {
    return vec_mat_1(other, target);
  }

  Vector& vec_mat_1(const Vector& other, Vector& target) const {
    target.reset();
    for (HighsInt col = 0; col < num_col; col++) {
      double dot = 0.0;
      for (HighsInt j = start[col]; j < start[col + 1]; j++) {
        dot += other.value[index[j]] * value[j];
      }
      target.value[col] = dot;
    }

    target.resparsify();
    return target;
  }

  Vector vec_mat(const Vector& other) const {
    Vector result(num_col);

    return vec_mat(other, result);
  }

  // computes this * mat, where "this" is a tranposed matrix
  MatrixBase tran_mat_(const MatrixBase& other) {
    MatrixBase res;
    res.num_row = num_col;
    res.num_col = other.num_col;

    res.start.push_back(0);
    Vector buffer_col(other.num_row);
    Vector buffer_col_res(num_col);
    for (HighsInt r = 0; r < other.num_col; r++) {
      other.extractcol(r, buffer_col);

      vec_mat(buffer_col, buffer_col_res);
      for (HighsInt i = 0; i < buffer_col_res.num_nz; i++) {
        res.index.push_back(buffer_col_res.index[i]);
        res.value.push_back(buffer_col_res.value[buffer_col_res.index[i]]);
      }
      res.start.push_back(res.start[r] + buffer_col_res.num_nz);
    }

    return res;
  }

  Vector& extractcol(HighsInt col, Vector& target) const {
    assert(target.dim == num_row);
    target.reset();

    if (col >= num_col) {
      target.index[0] = col - num_col;
      target.value[col - num_col] = 1.0;
      target.num_nz = 1;
    } else {
      for (HighsInt i = 0; i < start[col + 1] - start[col]; i++) {
        target.index[i] = index[start[col] + i];
        target.value[target.index[i]] = value[start[col] + i];
      }
      target.num_nz = start[col + 1] - start[col];
    }

    return target;
  }

  Vector extractcol(HighsInt col) const {
    Vector res(num_row);

    return extractcol(col, res);
  }
};

struct Matrix {
 private:
  MatrixBase tran;
  bool has_transpose = false;

  void transpose() {
    if (!has_transpose) {
      std::vector<std::vector<HighsInt>> row_indices(mat.num_row);
      std::vector<std::vector<double>> row_values(mat.num_row);

      for (HighsInt col = 0; col < mat.num_col; col++) {
        for (HighsInt entry = mat.start[col]; entry < mat.start[col + 1];
             entry++) {
          HighsInt row = mat.index[entry];
          double val = mat.value[entry];
          row_indices[row].push_back(col);
          row_values[row].push_back(val);
        }
      }
      tran.start.clear();
      tran.index.clear();
      tran.value.clear();
      tran.start.reserve(mat.num_row + 1);
      tran.index.reserve(mat.index.size());
      tran.value.reserve(mat.value.size());

      tran.start.push_back(0);
      for (HighsInt row = 0; row < mat.num_row; row++) {
        tran.index.insert(tran.index.end(), row_indices[row].begin(),
                          row_indices[row].end());
        tran.value.insert(tran.value.end(), row_values[row].begin(),
                          row_values[row].end());

        tran.start.push_back(tran.start[row] + row_indices[row].size());
      }

      tran.num_col = mat.num_row;
      tran.num_row = mat.num_col;
    }
  }

 public:
  MatrixBase mat;

  Matrix(HighsInt nr, HighsInt nc) {
    mat.num_row = nr;
    mat.num_col = nc;
  };

  Matrix(const MatrixBase& m, bool needstran) {
    mat = m;
    // if (needstran) {
    //    transpose();
    // }
  }

  void append(const Vector& vec) {
    if (mat.num_col == 0 && mat.start.size() == 0) {
      mat.start.push_back(0);
    }
    for (HighsInt i = 0; i < vec.num_nz; i++) {
      mat.index.push_back(vec.index[i]);
      mat.value.push_back(vec.value[vec.index[i]]);
    }
    mat.start.push_back(mat.start[mat.num_col] + vec.num_nz);
    mat.num_col++;
    has_transpose = false;
  }

  void append(HighsInt* idx, double* val, HighsInt nnz) {
    if (mat.num_col == 0 && mat.start.size() == 0) {
      mat.start.push_back(0);
    }
    for (HighsInt i = 0; i < nnz; i++) {
      mat.index.push_back(idx[i]);
      mat.value.push_back(val[i]);
    }
    mat.start.push_back(mat.start[mat.num_col] + nnz);
    mat.num_col++;
    has_transpose = false;
  }

  void append(HighsInt num_nz, HighsInt* index, double* value) {
    if (mat.num_col == 0 && mat.start.size() == 0) {
      mat.start.push_back(0);
    }
    for (HighsInt i = 0; i < num_nz; i++) {
      mat.index.push_back(index[i]);
      mat.value.push_back(value[i]);
    }
    mat.start.push_back(mat.start[mat.num_col] + num_nz);
    mat.num_col++;
    has_transpose = false;
  }

  void dropcol(HighsInt col) {
    assert(col < mat.num_col);
    has_transpose = false;

    mat.index.erase(mat.index.begin() + mat.start[col],
                    mat.index.begin() + mat.start[col + 1]);
    mat.value.erase(mat.value.begin() + mat.start[col],
                    mat.value.begin() + mat.start[col + 1]);

    HighsInt num_elements_in_col = mat.start[col + 1] - mat.start[col];
    for (; col < mat.num_col; col++) {
      mat.start[col] = mat.start[col + 1] - num_elements_in_col;
    }
    mat.start.pop_back();
    mat.num_col--;
  }

  MatrixBase& t() {
    if (!has_transpose) {
      transpose();
      has_transpose = true;
    }
    return tran;
  }

  Matrix mat_mat(Matrix& other) {
    Matrix res(mat.num_row, 0);

    Vector buffer(other.mat.num_row);
    Vector buffer2(mat.num_col);
    for (HighsInt col = 0; col < other.mat.num_col; col++) {
      res.append(vec_mat(other.mat.extractcol(col, buffer), buffer2));
    }

    return res;
  }

  Matrix tran_mat(Matrix& other) {
    Matrix res(mat.num_col, 0);

    Vector buffer(other.mat.num_row);
    Vector buffer2(mat.num_row);
    for (HighsInt col = 0; col < other.mat.num_col; col++) {
      res.append(mat_vec(other.mat.extractcol(col, buffer), buffer2));
    }
    return res;
  }

  Vector& mat_vec(const Vector& other, Vector& target) {
    return mat.mat_vec(other, target);
  }

  Vector mat_vec(const Vector& other) { return mat.mat_vec(other); }

  Vector vec_mat(const Vector& other) const { return mat.vec_mat(other); }

  Vector& vec_mat(const Vector& other, Vector& target) const {
    return mat.vec_mat(other, target);
  }

  Vector vec_mat(HighsInt* index, double* value, HighsInt num_nz) {
    return mat.vec_mat(index, value, num_nz);
  }

  void report(std::string name = "") const {
    if (name != "") {
      printf("%s:", name.c_str());
    }
    printf("[%" HIGHSINT_FORMAT " x %" HIGHSINT_FORMAT "]\n", mat.num_row,
           mat.num_col);
    printf("start: ");
    for (HighsInt i : mat.start) {
      printf("%" HIGHSINT_FORMAT " ", i);
    }
    printf("\n");

    printf("index: ");
    for (HighsInt i : mat.index) {
      printf("%" HIGHSINT_FORMAT " ", i);
    }
    printf("\n");

    printf("value: ");
    for (double d : mat.value) {
      printf("%lf ", d);
    }
    printf("\n");
  }
};

#endif
