import React from 'react';
import {Input, Text, Button} from '@rneui/themed';
import {View, StyleSheet} from 'react-native';

const LoginScreen = () => {
  return (
    <View style={styles.container}>
      <Input label="Usuario" placeholder="Usuario" />
      <Input label="Contraseña" placeholder="Contraseña" />

      <Button title="Log In" />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    marginHorizontal: 20,
    justifyContent: 'center',
  },
});

export default LoginScreen;
