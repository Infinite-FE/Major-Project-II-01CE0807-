import 'package:deepfakedetector/analysis_result.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

class AnalysisDetailsScreen extends StatelessWidget {
  final AnalysisResult result;
  const AnalysisDetailsScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    final isFake = result.isDeepfake;
    final color = isFake ? Colors.red.shade400 : Colors.green.shade400;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Analysis Details'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(result.fileName, style: textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Text(DateFormat.yMMMd().add_jm().format(result.dateTime), style: textTheme.titleMedium),
            const SizedBox(height: 24),
            Card(
              color: color,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(isFake ? Icons.gpp_bad_outlined : Icons.verified_user_outlined, color: Colors.white, size: 30),
                    const SizedBox(width: 12),
                    Text(
                      isFake ? 'Deepfake Detected' : 'Authentic Audio',
                      style: textTheme.headlineSmall?.copyWith(color: Colors.white, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            Text('Confidence: ${result.confidence.toStringAsFixed(2)}%', style: textTheme.titleLarge),
            const SizedBox(height: 16),
            Text('Probabilities:', style: textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Text('  - Fake: ${result.probabilities['fake']?.toStringAsFixed(2)}%', style: textTheme.bodyLarge),
            Text('  - Real: ${result.probabilities['real']?.toStringAsFixed(2)}%', style: textTheme.bodyLarge),
          ],
        ),
      ),
    );
  }
}
