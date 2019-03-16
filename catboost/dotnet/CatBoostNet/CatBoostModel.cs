using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace CatBoostNet
{
    public class CatBoostModel : ITransformer
    {
        public bool IsRowToRowMapper => throw new NotImplementedException();

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            throw new NotImplementedException();
        }

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        {
            throw new NotImplementedException();
        }

        public void Save(ModelSaveContext ctx)
        {
            throw new NotImplementedException();
        }

        public IDataView Transform(IDataView input)
        {
            throw new NotImplementedException();
        }
    }
}
