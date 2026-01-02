import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:deepfakedetector/analysis_details_screen.dart';
import 'package:deepfakedetector/analysis_result.dart';
import 'package:deepfakedetector/api_service.dart';
import 'package:deepfakedetector/auth_service.dart';
import 'package:deepfakedetector/education_screen.dart';
import 'package:deepfakedetector/history_screen.dart';
import 'package:deepfakedetector/login_screen.dart';
import 'package:file_picker/file_picker.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Deepfake Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        // Using a more sophisticated color palette
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF6750A4),
          brightness: Brightness.light,
        ),
      ),
      home: const AuthWrapper(),
    );
  }
}

// ... Keep AuthWrapper as is ...
class AuthWrapper extends StatelessWidget {
  const AuthWrapper({super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder(
      stream: AuthService().authStateChanges,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Scaffold(body: Center(child: CircularProgressIndicator()));
        }
        return snapshot.hasData ? const MyHomePage() : const LoginScreen();
      },
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});
  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _selectedIndex = 0;
  final List<AnalysisResult> _history = [];
  AnalysisResult? _lastResult;
  bool _isAnalyzing = false;
  FlutterSoundRecorder? _recorder;
  bool _isRecording = false;

  @override
  void initState() {
    super.initState();
    _recorder = FlutterSoundRecorder();
    _recorder!.openRecorder();
    _loadHistoryFromDatabase();
  }

  // --- LOGIC REMAINS THE SAME AS YOUR WORKING VERSION ---
  Future<void> _loadHistoryFromDatabase() async {
    final user = AuthService().currentUser;
    if (user == null) return;
    final snapshot = await FirebaseFirestore.instance
        .collection('analysis_history')
        .where('userId', isEqualTo: user.uid)
        .orderBy('createdAt', descending: true)
        .get();

    final List<AnalysisResult> loadedHistory = snapshot.docs.map((doc) {
      final data = doc.data();
      return AnalysisResult(
        fileName: data['fileName'],
        filePath: data['filePath'],
        dateTime: (data['createdAt'] as Timestamp).toDate(),
        label: data['label'],
        confidence: (data['confidence'] as num).toDouble(),
        probabilities: Map<String, dynamic>.from(data['probabilities']),
        isDeepfake: data['label'] == 'fake',
      );
    }).toList();

    setState(() {
      _history.clear();
      _history.addAll(loadedHistory);
      _lastResult = _history.isNotEmpty ? _history.first : null;
    });
  }

  Future<void> _saveAnalysisToDatabase(AnalysisResult result) async {
    final user = AuthService().currentUser;
    if (user == null) return;
    await FirebaseFirestore.instance.collection('analysis_history').add({
      'userId': user.uid,
      'fileName': result.fileName,
      'filePath': result.filePath,
      'label': result.label,
      'confidence': result.confidence,
      'probabilities': result.probabilities,
      'createdAt': FieldValue.serverTimestamp(),
    });
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      final filePath = await _recorder!.stopRecorder();
      setState(() => _isRecording = false);
      if (filePath != null) await _analyzeAudioFile(filePath, 'recorded_audio.wav');
    } else {
      final micPermission = await Permission.microphone.request();
      if (!micPermission.isGranted) return;
      final tempDir = await getTemporaryDirectory();
      final filePath = path.join(tempDir.path, 'temp_audio.wav');
      await _recorder!.startRecorder(toFile: filePath, codec: Codec.pcm16WAV);
      setState(() => _isRecording = true);
    }
  }

  Future<void> _importAudio() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.audio);
    if (result != null && result.files.single.path != null) {
      await _analyzeAudioFile(result.files.single.path!, result.files.single.name);
    }
  }

  Future<void> _analyzeAudioFile(String filePath, String fileName) async {
    setState(() { _isAnalyzing = true; _lastResult = null; });
    try {
      final apiResult = await ApiService.predictAudio(filePath);
      final result = AnalysisResult(
        fileName: fileName,
        filePath: filePath,
        dateTime: DateTime.now(),
        label: apiResult['label'],
        confidence: (apiResult['confidence'] as num).toDouble(),
        probabilities: apiResult['probabilities'],
        isDeepfake: apiResult['label'] == 'fake',
      );
      setState(() { _history.insert(0, result); _lastResult = result; });
      await _saveAnalysisToDatabase(result);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Analysis failed')));
    } finally {
      setState(() => _isAnalyzing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: Text(['Voice Shield', 'History', 'Learn'][_selectedIndex],
            style: const TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          IconButton(icon: const Icon(Icons.logout_rounded), onPressed: () => AuthService().logout()),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Theme.of(context).colorScheme.primaryContainer.withOpacity(0.4),
              Theme.of(context).colorScheme.surface,
            ],
          ),
        ),
        child: IndexedStack(
          index: _selectedIndex,
          children: [
            DetectionScreen(
              lastResult: _lastResult,
              isAnalyzing: _isAnalyzing,
              isRecording: _isRecording,
              onImportAudio: _importAudio,
              onToggleRecording: _toggleRecording,
            ),
            HistoryScreen(history: _history),
            const EducationScreen(),
          ],
        ),
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (i) => setState(() => _selectedIndex = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.shield_outlined), selectedIcon: Icon(Icons.shield), label: 'Detector'),
          NavigationDestination(icon: Icon(Icons.history_rounded), label: 'History'),
          NavigationDestination(icon: Icon(Icons.school_outlined), label: 'Education'),
        ],
      ),
    );
  }
}

class DetectionScreen extends StatelessWidget {
  final AnalysisResult? lastResult;
  final bool isAnalyzing;
  final bool isRecording;
  final VoidCallback onImportAudio;
  final VoidCallback onToggleRecording;

  const DetectionScreen({
    super.key, this.lastResult, required this.isAnalyzing,
    required this.isRecording, required this.onImportAudio, required this.onToggleRecording,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const SizedBox(height: 100),
          Expanded(
            child: Center(
              child: isAnalyzing
                  ? const Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(strokeWidth: 6),
                  SizedBox(height: 20),
                  Text("Analyzing voice patterns...", style: TextStyle(fontWeight: FontWeight.w500)),
                ],
              )
                  : lastResult != null
                  ? AnalysisResultCard(result: lastResult!)
                  : _buildEmptyState(context),
            ),
          ),
          _buildActionSection(context),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  Widget _buildEmptyState(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(Icons.multitrack_audio_rounded, size: 80, color: Theme.of(context).colorScheme.primary.withOpacity(0.5)),
        const SizedBox(height: 16),
        const Text('Ready to Verify', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
        const Text('Upload or record audio to check for deepfakes', textAlign: TextAlign.center),
      ],
    );
  }

  Widget _buildActionSection(BuildContext context) {
    return Column(
      children: [
        GestureDetector(
          onTap: isAnalyzing ? null : onToggleRecording,
          child: Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: isRecording ? Colors.red.withOpacity(0.2) : Theme.of(context).colorScheme.primaryContainer,
              shape: BoxShape.circle,
            ),
            child: Container(
              height: 80,
              width: 80,
              decoration: BoxDecoration(
                color: isRecording ? Colors.red : Theme.of(context).colorScheme.primary,
                shape: BoxShape.circle,
                boxShadow: [BoxShadow(color: (isRecording ? Colors.red : Colors.blue).withOpacity(0.3), blurRadius: 15, spreadRadius: 5)],
              ),
              child: Icon(isRecording ? Icons.stop_rounded : Icons.mic_rounded, color: Colors.white, size: 40),
            ),
          ),
        ),
        const SizedBox(height: 30),
        OutlinedButton.icon(
          onPressed: isAnalyzing || isRecording ? null : onImportAudio,
          icon: const Icon(Icons.upload_file_rounded),
          label: const Text('Import Audio File'),
          style: OutlinedButton.styleFrom(
            minimumSize: const Size(200, 50),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
        ),
      ],
    );
  }
}

class AnalysisResultCard extends StatefulWidget {
  final AnalysisResult result;
  const AnalysisResultCard({super.key, required this.result});

  @override
  State<AnalysisResultCard> createState() => _AnalysisResultCardState();
}

class _AnalysisResultCardState extends State<AnalysisResultCard> {
  final FlutterSoundPlayer _player = FlutterSoundPlayer();
  bool _isPlaying = false;

  @override
  void initState() { super.initState(); _player.openPlayer(); }
  @override
  void dispose() { _player.closePlayer(); super.dispose(); }

  Future<void> _togglePlayback() async {
    if (_isPlaying) {
      await _player.stopPlayer();
      setState(() => _isPlaying = false);
    } else {
      await _player.startPlayer(fromURI: widget.result.filePath, whenFinished: () => setState(() => _isPlaying = false));
      setState(() => _isPlaying = true);
    }
  }

  @override
  Widget build(BuildContext context) {
    final bool isFake = widget.result.isDeepfake;
    final color = isFake ? Colors.red : Colors.green;

    return Card(
      elevation: 0,
      color: color.withOpacity(0.05),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(24),
        side: BorderSide(color: color.withOpacity(0.2), width: 2),
      ),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(color: color.withOpacity(0.1), shape: BoxShape.circle),
              child: Icon(isFake ? Icons.warning_amber_rounded : Icons.check_circle_outline_rounded, size: 48, color: color),
            ),
            const SizedBox(height: 16),
            Text(
              isFake ? 'Deepfake Detected' : 'Authentic Voice',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: color),
            ),
            const SizedBox(height: 24),
            // Confidence Bar
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text("Confidence Level", style: TextStyle(fontWeight: FontWeight.w500)),
                    Text("${widget.result.confidence.toStringAsFixed(1)}%"),
                  ],
                ),
                const SizedBox(height: 8),
                LinearProgressIndicator(
                  value: widget.result.confidence / 100,
                  backgroundColor: color.withOpacity(0.1),
                  color: color,
                  borderRadius: BorderRadius.circular(10),
                  minHeight: 8,
                ),
              ],
            ),
            const SizedBox(height: 24),
            Material(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              child: ListTile(
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                leading: IconButton.filledTonal(
                  icon: Icon(_isPlaying ? Icons.pause_rounded : Icons.play_arrow_rounded),
                  onPressed: _togglePlayback,
                ),
                title: Text(widget.result.fileName, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 14)),
                subtitle: const Text("Tap to play recorded audio", style: TextStyle(fontSize: 12)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}