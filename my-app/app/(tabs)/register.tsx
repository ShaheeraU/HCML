import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, TextInput, Alert, ActivityIndicator, ScrollView, Image } from 'react-native';
import { router } from 'expo-router';
import * as ImagePicker from 'expo-image-picker';
import { Ionicons } from '@expo/vector-icons';
import { API_URL } from '@/constants/api';

interface ImageAsset {
  uri: string;
  base64?: string;
  width: number;
  height: number;
}

export default function RegisterScreen() {
  const [name, setName] = useState('');
  const [image, setImage] = useState<ImageAsset | null>(null);
  const [loading, setLoading] = useState(false);

  const takePhoto = async () => {
    try {
      const camPerm = await ImagePicker.requestCameraPermissionsAsync();
      if (camPerm.status !== 'granted') {
        Alert.alert('Permission required', 'Camera permission is needed to take a photo.');
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true, // enables OS crop/align UI
        aspect: [1, 1], // square crop for face
        base64: true,
        quality: 0.9,
      });
      

      if (!result.canceled && result.assets[0]) {
        setImage(result.assets[0] as ImageAsset);
      }
    } catch (error) {
        
      Alert.alert('Error', 'Failed to take photo');
    }
  };

  const registerFace = async () => {
    if (!name.trim()) {
      Alert.alert('Error', 'Please enter a name');
      return;
    }

    if (!image || !image.base64) {
      Alert.alert('Error', 'Please take a photo');
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: name.trim(),
          image: image.base64,
        }),
      });

      const result = await response.json();

      if (response.ok) {
        Alert.alert(
          'Success!',
          `Face registered successfully!\nUser ID: ${result.user_id}`,
          [
            {
              text: 'OK',
              onPress: () => router.push('/(tabs)/welcome'),
            },
          ]
        );
      } else {
        Alert.alert('Error', result.error || 'Registration failed');
      }
    } catch (error) {
      Alert.alert('Error', 'Network error. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={24} color="#2196F3" />
        </TouchableOpacity>
        <Text style={styles.title}>Register Face</Text>
      </View>

      <View style={styles.inputContainer}>
        <Text style={styles.label}>Enter Your Name:</Text>
        <TextInput
          style={styles.input}
          value={name}
          onChangeText={setName}
          placeholder="Enter your name"
          placeholderTextColor="#999"
        />
      </View>

      <View style={styles.imageContainer}>
        {!image ? (
          <View style={styles.placeholderContainer}>
            <Ionicons name="person" size={80} color="#ccc" />
            <Text style={styles.placeholderText}>No image captured</Text>
            <TouchableOpacity style={[styles.button, styles.selectButton]} onPress={takePhoto}>
              <Ionicons name="camera" size={24} color="white" style={styles.buttonIcon} />
              <Text style={styles.buttonText}>Take Photo</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <View style={styles.imagePreview}>
            <Image source={{ uri: image.uri }} style={styles.previewImage} />
            <View style={styles.imageActions}>
              <TouchableOpacity style={styles.retakeButton} onPress={() => setImage(null)}>
                <Text style={styles.retakeButtonText}>Retake</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.selectAnotherButton} onPress={takePhoto}>
                <Text style={styles.selectAnotherButtonText}>Retake & Crop</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </View>

      {image && (
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[styles.button, styles.registerButton]}
            onPress={registerFace}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="white" />
            ) : (
              <>
                <Ionicons name="checkmark" size={24} color="white" style={styles.buttonIcon} />
                <Text style={styles.buttonText}>Register Face</Text>
              </>
            )}
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    paddingTop: 60,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  backButton: {
    marginRight: 15,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  inputContainer: {
    padding: 20,
  },
  label: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
    backgroundColor: 'white',
  },
  imageContainer: {
    height: 400,
    margin: 20,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: 'white',
  },
  placeholderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  placeholderText: {
    fontSize: 18,
    color: '#999',
    marginBottom: 30,
    textAlign: 'center',
  },
  selectButton: {
    backgroundColor: '#2196F3',
  },
  imagePreview: {
    flex: 1,
    padding: 20,
  },
  previewImage: {
    width: '100%',
    height: 300,
    borderRadius: 12,
    marginBottom: 20,
  },
  imageActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  retakeButton: {
    backgroundColor: '#ff9800',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 20,
  },
  retakeButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  selectAnotherButton: {
    backgroundColor: '#2196F3',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 20,
  },
  selectAnotherButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  buttonContainer: {
    padding: 20,
  },
  button: {
    flexDirection: 'row',
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 25,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  registerButton: {
    backgroundColor: '#9C27B0',
  },
  buttonIcon: {
    marginRight: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
});
