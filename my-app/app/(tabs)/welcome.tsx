import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { router } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

export default function WelcomeScreen() {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Face Recognition App</Text>
        <Text style={styles.subtitle}>Secure authentication using facial recognition</Text>
      </View>

      <View style={styles.logoContainer}>
        <Ionicons name="person-circle" size={120} color="#4CAF50" />
        <Text style={styles.logoText}>AI-Powered</Text>
        <Text style={styles.logoText}>Face Recognition</Text>
      </View>

      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[styles.button, styles.registerButton]}
          onPress={() => router.push('/(tabs)/register')}
        >
          <Ionicons name="person-add" size={24} color="white" style={styles.buttonIcon} />
          <Text style={styles.buttonText}>Register New Face</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, styles.loginButton]}
          onPress={() => router.push('/(tabs)/login')}
        >
          <Ionicons name="log-in" size={24} color="white" style={styles.buttonIcon} />
          <Text style={styles.buttonText}>Login with Face</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.footer}>
        <Text style={styles.footerText}>Powered by MobileFaceNet</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginTop: 60,
    marginBottom: 40,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  logoContainer: {
    alignItems: 'center',
    marginBottom: 60,
  },
  logoText: {
    fontSize: 18,
    color: '#666',
    marginBottom: 5,
  },
  buttonContainer: {
    gap: 20,
    marginBottom: 40,
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
    backgroundColor: '#4CAF50',
  },
  loginButton: {
    backgroundColor: '#2196F3',
  },
  buttonIcon: {
    marginRight: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  footer: {
    position: 'absolute',
    bottom: 30,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  footerText: {
    color: '#999',
    fontSize: 14,
  },
});
