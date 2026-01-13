using Microsoft.Extensions.VectorData;

namespace RAGMovieApp
{
    /// <summary>
    /// Represents a document chunk extracted from a PDF file
    /// </summary>
    public class Document
    {
        [VectorStoreRecordKey]
        public Guid Key { get; set; } = Guid.NewGuid();

        [VectorStoreRecordData]
        public string Title { get; set; } = null!;

        [VectorStoreRecordData]
        public string FileName { get; set; } = null!;

        [VectorStoreRecordData]
        public int PageNumber { get; set; }

        [VectorStoreRecordData]
        public int ChunkIndex { get; set; }

        [VectorStoreRecordData]
        public string Content { get; set; } = null!;

        [VectorStoreRecordVector(768, DistanceFunction = DistanceFunction.CosineSimilarity)]
        public ReadOnlyMemory<float>? ContentEmbedding { get; set; }
    }
}
