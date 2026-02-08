# DexteraAI Mobile (Flutter)

## Status: Planned

The mobile app will use the same gesture recognition pipeline via TensorFlow Lite.

### Architecture

```
Flutter App
├── lib/
│   ├── main.dart
│   ├── screens/
│   │   ├── home_screen.dart
│   │   ├── camera_screen.dart
│   │   └── settings_screen.dart
│   ├── services/
│   │   ├── gesture_service.dart      # TFLite inference
│   │   ├── camera_service.dart       # Camera access
│   │   └── landmark_service.dart     # MediaPipe Hands
│   ├── models/
│   │   ├── gesture_result.dart
│   │   └── hand_landmarks.dart
│   └── widgets/
│       ├── camera_preview.dart
│       ├── landmark_overlay.dart
│       └── gesture_display.dart
├── assets/
│   └── models/
│       └── gesture.tflite            # Exported model
└── pubspec.yaml
```

### Dependencies
- `camera` — Camera access
- `tflite_flutter` — TFLite inference
- `google_mlkit_commons` — MediaPipe integration
- `provider` — State management

### To Initialize
```bash
cd apps/mobile
flutter create . --org ai.dextera
flutter pub add camera tflite_flutter provider
```

### Privacy
Same as all DexteraAI apps:
- All inference on-device
- No images/video transmitted
- No cloud dependency
