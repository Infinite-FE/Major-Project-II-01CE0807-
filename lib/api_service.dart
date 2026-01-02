import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // NOTE: This is the standard address for a localhost server when running from an Android emulator.
  // If your backend is hosted elsewhere or you're using an iOS simulator (which uses localhost),
  // you will need to change this URL.
  static const String _baseUrl = 'http://10.0.2.2:8000';

  static Future<Map<String, dynamic>> predictAudio(String filePath) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/predict'),
      );

      // Add the file to the request
      request.files.add(await http.MultipartFile.fromPath('file', filePath));

      // Send the request
      final streamedResponse = await request.send();

      // Get the response
      if (streamedResponse.statusCode == 200) {
        final response = await http.Response.fromStream(streamedResponse);
        final Map<String, dynamic> data = json.decode(response.body);
        return data;
      } else {
        final response = await http.Response.fromStream(streamedResponse);
        throw Exception(
            'Failed to get prediction. Status code: ${streamedResponse.statusCode}\nResponse: ${response.body}');
      }
    } catch (e) {
      throw Exception('Error during prediction: $e');
    }
  }
}
