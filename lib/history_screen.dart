import 'package:deepfakedetector/analysis_details_screen.dart';
import 'package:deepfakedetector/analysis_result.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

class HistoryScreen extends StatelessWidget {
  final List<AnalysisResult> history;
  const HistoryScreen({super.key, required this.history});

  @override
  Widget build(BuildContext context) {
    if (history.isEmpty) {
      return const Center(child: Text('No analysis history yet.'));
    }

    return ListView.builder(
      itemCount: history.length,
      itemBuilder: (context, index) {
        final item = history[index];
        final color = item.isDeepfake ? Colors.red.shade400 : Colors.green.shade400;

        return Card(
          margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: ListTile(
            leading: Icon(
              item.isDeepfake ? Icons.gpp_bad_outlined : Icons.verified_user_outlined,
              color: color,
              size: 40,
            ),
            title: Text(item.fileName, style: const TextStyle(fontWeight: FontWeight.bold)),
            subtitle: Text(
              '${item.label.toUpperCase()} (${item.confidence.toStringAsFixed(1)}%)\n${DateFormat.yMMMd().add_jm().format(item.dateTime)}',
            ),
            isThreeLine: true,
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => AnalysisDetailsScreen(result: item),
                ),
              );
            },
          ),
        );
      },
    );
  }
}
