using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;

namespace RAGMovieApp
{
    /// <summary>
    /// Service for extracting text content from PDF files
    /// </summary>
    public static class PdfExtractor
    {
        /// <summary>
        /// Default chunk size in characters for splitting PDF content
        /// </summary>
        private const int DefaultChunkSize = 1000;

        /// <summary>
        /// Default overlap between chunks to maintain context
        /// </summary>
        private const int DefaultOverlap = 200;

        /// <summary>
        /// Extracts documents from a PDF file, splitting content into chunks
        /// </summary>
        /// <param name="pdfPath">Path to the PDF file</param>
        /// <param name="chunkSize">Maximum characters per chunk</param>
        /// <param name="overlap">Characters to overlap between chunks</param>
        /// <returns>List of Document objects representing chunks of the PDF</returns>
        public static List<Document> ExtractFromPdf(string pdfPath, int chunkSize = DefaultChunkSize, int overlap = DefaultOverlap)
        {
            var documents = new List<Document>();
            var fileName = Path.GetFileName(pdfPath);
            var title = Path.GetFileNameWithoutExtension(pdfPath);

            using var document = PdfDocument.Open(pdfPath);

            foreach (var page in document.GetPages())
            {
                var pageText = page.Text;

                if (string.IsNullOrWhiteSpace(pageText))
                    continue;

                // Clean up the text
                pageText = CleanText(pageText);

                // Split page text into chunks
                var chunks = SplitIntoChunks(pageText, chunkSize, overlap);

                for (int i = 0; i < chunks.Count; i++)
                {
                    documents.Add(new Document
                    {
                        Title = title,
                        FileName = fileName,
                        PageNumber = page.Number,
                        ChunkIndex = i,
                        Content = chunks[i]
                    });
                }
            }

            return documents;
        }

        /// <summary>
        /// Extracts documents from all PDF files in a directory
        /// </summary>
        /// <param name="directoryPath">Path to the directory containing PDFs</param>
        /// <param name="chunkSize">Maximum characters per chunk</param>
        /// <param name="overlap">Characters to overlap between chunks</param>
        /// <returns>List of all Document objects from all PDFs</returns>
        public static List<Document> ExtractFromDirectory(string directoryPath, int chunkSize = DefaultChunkSize, int overlap = DefaultOverlap)
        {
            var allDocuments = new List<Document>();
            var pdfFiles = Directory.GetFiles(directoryPath, "*.pdf", SearchOption.AllDirectories);

            foreach (var pdfFile in pdfFiles)
            {
                try
                {
                    var documents = ExtractFromPdf(pdfFile, chunkSize, overlap);
                    allDocuments.AddRange(documents);
                    Console.WriteLine($"Extracted {documents.Count} chunks from: {Path.GetFileName(pdfFile)}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing {Path.GetFileName(pdfFile)}: {ex.Message}");
                }
            }

            return allDocuments;
        }

        /// <summary>
        /// Cleans text by normalizing whitespace and removing unwanted characters
        /// </summary>
        private static string CleanText(string text)
        {
            // Replace multiple whitespace with single space
            text = System.Text.RegularExpressions.Regex.Replace(text, @"\s+", " ");

            // Remove control characters except newlines
            text = System.Text.RegularExpressions.Regex.Replace(text, @"[\x00-\x08\x0B\x0C\x0E-\x1F]", "");

            return text.Trim();
        }

        /// <summary>
        /// Splits text into overlapping chunks for better context preservation
        /// </summary>
        private static List<string> SplitIntoChunks(string text, int chunkSize, int overlap)
        {
            var chunks = new List<string>();

            if (text.Length <= chunkSize)
            {
                chunks.Add(text);
                return chunks;
            }

            int start = 0;
            while (start < text.Length)
            {
                int length = Math.Min(chunkSize, text.Length - start);
                var chunk = text.Substring(start, length);

                // Try to break at sentence or word boundary
                if (start + length < text.Length)
                {
                    var lastSentenceEnd = FindLastSentenceEnd(chunk);
                    if (lastSentenceEnd > chunkSize / 2)
                    {
                        chunk = chunk.Substring(0, lastSentenceEnd + 1);
                    }
                }

                chunks.Add(chunk.Trim());

                // Move start position, accounting for overlap
                start += chunk.Length - overlap;

                // Avoid infinite loop for small texts
                if (chunk.Length <= overlap)
                    break;
            }

            return chunks;
        }

        /// <summary>
        /// Finds the last sentence ending in a text chunk
        /// </summary>
        private static int FindLastSentenceEnd(string text)
        {
            var sentenceEnders = new[] { '.', '!', '?' };
            int lastIndex = -1;

            foreach (var ender in sentenceEnders)
            {
                int index = text.LastIndexOf(ender);
                if (index > lastIndex)
                    lastIndex = index;
            }

            return lastIndex;
        }
    }
}
