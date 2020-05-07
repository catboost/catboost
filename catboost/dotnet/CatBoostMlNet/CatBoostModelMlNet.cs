using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using CatBoostNet;

namespace CatBoostMlNet {
    /// <summary>
    /// High-level, ML.NET-compliant class for making predictions using CatBoost models.
    /// </summary>
    public class CatBoostModelMlNet : ITransformer, IDisposable {
        private CatBoostModel Model { get; }

        /// <summary>
        /// Model constructor
        /// </summary>
        /// <param name="modelFilePath">Path to the model file</param>
        /// <param name="targetFeature">Name of the target feature</param>
        public CatBoostModelMlNet(string modelFilePath, string targetFeature) {
            Model = new CatBoostModel(modelFilePath, targetFeature);
        }

        /// <inheritdoc />
        public bool IsRowToRowMapper => true;

        /// <inheritdoc />
        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) {
            throw new NotImplementedException();
        }

        /// <inheritdoc />
        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema) {
            throw new NotImplementedException();
        }

        /// <inheritdoc />
        public void Save(ModelSaveContext ctx) {
            // Nothing to do here, we are loading model from the file anyway ;)
        }

        /// <inheritdoc />
        public IDataView Transform(IDataView input) => Model.Transform(input);

        /// <summary>
        /// Dispose of unmanaged resources
        /// </summary>
        public void Dispose() {
            Model?.Dispose();
        }
    }
}
