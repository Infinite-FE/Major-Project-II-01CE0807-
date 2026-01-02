import 'package:flutter/material.dart';

class EducationScreen extends StatelessWidget {
  const EducationScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      // Transparent background to show the home screen gradient if needed
      backgroundColor: Colors.transparent,
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 100, 20, 20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(theme),
            const SizedBox(height: 24),

            _buildInfoCard(
              theme,
              title: 'What are Deepfakes?',
              icon: Icons.psychology_rounded,
              content: 'Deepfakes are synthetic media where a person\'s likeness or voice is replaced using AI. Audio deepfakes (voice cloning) can convincingly mimic anyone, making them a powerful tool for both creativity and deception.',
              color: theme.colorScheme.primary,
            ),

            const SizedBox(height: 16),
            const Text(
              'Red Flags to Listen For',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),

            _buildDetectionTile(
              context,
              icon: Icons.graphic_eq_rounded,
              title: 'Unnatural Rhythm',
              subtitle: 'Listen for strange pauses or perfectly consistent timing that lacks human "breath."',
            ),
            _buildDetectionTile(
              context,
              icon: Icons.face_retouching_off_rounded,
              title: 'Flat Emotion',
              subtitle: 'AI often struggles to replicate genuine anger, excitement, or subtle sarcasm.',
            ),
            _buildDetectionTile(
              context,
              icon: Icons.blur_on_rounded,
              title: 'Digital Artifacts',
              subtitle: 'Metallic "chirping" sounds or inconsistent background noise often hide in the audio.',
            ),

            const SizedBox(height: 32),
            _buildProtectionTip(theme),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(ThemeData theme) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Educational Guide',
          style: theme.textTheme.displaySmall?.copyWith(
            fontWeight: FontWeight.bold,
            color: theme.colorScheme.onSurface,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          'Learn how to stay safe in the age of AI.',
          style: theme.textTheme.bodyLarge?.copyWith(color: Colors.grey[600]),
        ),
      ],
    );
  }

  Widget _buildInfoCard(ThemeData theme, {required String title, required IconData icon, required String content, required Color color}) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [color.withOpacity(0.8), color],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.3),
            blurRadius: 12,
            offset: const Offset(0, 6),
          )
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: Colors.white, size: 28),
              const SizedBox(width: 12),
              Text(
                title,
                style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            content,
            style: const TextStyle(color: Colors.white, fontSize: 15, height: 1.5),
          ),
        ],
      ),
    );
  }

  Widget _buildDetectionTile(BuildContext context, {required IconData icon, required String title, required String subtitle}) {
    return Card(
      elevation: 0,
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: Colors.grey.withOpacity(0.2)),
      ),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        leading: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.primary.withOpacity(0.1),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, color: Theme.of(context).colorScheme.primary),
        ),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text(subtitle, style: const TextStyle(fontSize: 13)),
      ),
    );
  }

  Widget _buildProtectionTip(ThemeData theme) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.amber.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.amber.withOpacity(0.5)),
      ),
      child: Row(
        children: [
          const Icon(Icons.lightbulb_outline_rounded, color: Colors.amber),
          const SizedBox(width: 16),
          Expanded(
            child: Text(
              'Pro Tip: Always verify suspicious voice notes by calling the person back on a trusted line.',
              style: TextStyle(color: Colors.amber[900], fontWeight: FontWeight.w500, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }
}