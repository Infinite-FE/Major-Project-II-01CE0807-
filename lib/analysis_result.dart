class AnalysisResult {
  final String fileName;
  final String filePath; // Add this
  final DateTime dateTime;
  final String label;
  final double confidence;
  final Map<String, dynamic> probabilities;
  final bool isDeepfake;

  AnalysisResult({
    required this.fileName,
    required this.filePath, // Add this
    required this.dateTime,
    required this.label,
    required this.confidence,
    required this.probabilities,
    required this.isDeepfake,
  });
}
