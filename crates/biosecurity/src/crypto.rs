//! Cryptographic operations and utilities
//! 
//! Implements the cryptographic backbone of the biological security framework,
//! supporting AES-256-GCM, RSA-4096, Ed25519, and SHA-3 algorithms.

use aes_gcm::{Aes256Gcm, Key, Nonce, aead::{Aead, AeadCore, KeyInit, OsRng}};
use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier};
use sha3::{Sha3_256, Sha3_512, Digest};
use hmac::{Hmac, Mac};
use pbkdf2::pbkdf2_hmac;
use ring::{rand::SystemRandom, signature::{RsaKeyPair, RSA_PKCS1_SHA256}};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::errors::{SecurityError, SecurityResult};
use crate::config::{CryptoConfig, HashAlgorithm};

/// AES-256-GCM symmetric encryption key
#[derive(Clone, ZeroizeOnDrop)]
pub struct SymmetricKey([u8; 32]);

impl SymmetricKey {
    /// Generate a new random symmetric key
    pub fn generate() -> Self {
        let key = Aes256Gcm::generate_key(&mut OsRng);
        Self(key.into())
    }

    /// Create key from bytes
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get key as bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Derive key from password using PBKDF2
    pub fn from_password(password: &[u8], salt: &[u8], iterations: u32) -> Self {
        let mut key = [0u8; 32];
        pbkdf2_hmac::<Sha3_256>(password, salt, iterations, &mut key);
        Self(key)
    }
}

/// Ed25519 key pair for digital signatures
#[derive(Clone)]
pub struct SignatureKeyPair {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
}

impl SignatureKeyPair {
    /// Generate a new key pair
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Create from seed bytes
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();
        
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Get the verifying (public) key
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.verifying_key
    }

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> Signature {
        self.signing_key.sign(message)
    }

    /// Verify a signature
    pub fn verify(&self, message: &[u8], signature: &Signature) -> SecurityResult<()> {
        self.verifying_key
            .verify(message, signature)
            .map_err(|e| SecurityError::Cryptographic(format!("Signature verification failed: {}", e)))
    }

    /// Export verifying key as bytes
    pub fn verifying_key_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }
}

/// Cryptographic context for security operations
pub struct CryptoContext {
    config: CryptoConfig,
    symmetric_key: SymmetricKey,
    signature_keypair: SignatureKeyPair,
    cipher: Aes256Gcm,
    rng: SystemRandom,
}

impl CryptoContext {
    /// Create new cryptographic context
    pub fn new(config: CryptoConfig) -> SecurityResult<Self> {
        let symmetric_key = SymmetricKey::generate();
        let signature_keypair = SignatureKeyPair::generate();
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(symmetric_key.as_bytes()));
        let rng = SystemRandom::new();

        Ok(Self {
            config,
            symmetric_key,
            signature_keypair,
            cipher,
            rng,
        })
    }

    /// Encrypt data using AES-256-GCM
    pub fn encrypt(&self, plaintext: &[u8], associated_data: Option<&[u8]>) -> SecurityResult<EncryptedData> {
        let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        
        let mut payload = aes_gcm::aead::Payload {
            msg: plaintext,
            aad: associated_data.unwrap_or(&[]),
        };

        let ciphertext = self.cipher
            .encrypt(&nonce, payload)
            .map_err(|e| SecurityError::Cryptographic(format!("Encryption failed: {}", e)))?;

        Ok(EncryptedData {
            ciphertext,
            nonce: nonce.into(),
            associated_data: associated_data.map(|ad| ad.to_vec()),
        })
    }

    /// Decrypt data using AES-256-GCM
    pub fn decrypt(&self, encrypted: &EncryptedData) -> SecurityResult<Vec<u8>> {
        let nonce = Nonce::from_slice(&encrypted.nonce);
        
        let payload = aes_gcm::aead::Payload {
            msg: &encrypted.ciphertext,
            aad: encrypted.associated_data.as_deref().unwrap_or(&[]),
        };

        let plaintext = self.cipher
            .decrypt(nonce, payload)
            .map_err(|e| SecurityError::Cryptographic(format!("Decryption failed: {}", e)))?;

        Ok(plaintext)
    }

    /// Sign data with Ed25519
    pub fn sign(&self, data: &[u8]) -> Signature {
        self.signature_keypair.sign(data)
    }

    /// Verify Ed25519 signature
    pub fn verify(&self, data: &[u8], signature: &Signature) -> SecurityResult<()> {
        self.signature_keypair.verify(data, signature)
    }

    /// Verify signature from external key
    pub fn verify_with_key(&self, data: &[u8], signature: &Signature, public_key: &[u8; 32]) -> SecurityResult<()> {
        let verifying_key = VerifyingKey::from_bytes(public_key)
            .map_err(|e| SecurityError::Cryptographic(format!("Invalid public key: {}", e)))?;
        
        verifying_key
            .verify(data, signature)
            .map_err(|e| SecurityError::Cryptographic(format!("Signature verification failed: {}", e)))
    }

    /// Compute hash using configured algorithm
    pub fn hash(&self, data: &[u8]) -> Vec<u8> {
        match self.config.hash_algorithm {
            HashAlgorithm::Sha3_256 => {
                let mut hasher = Sha3_256::new();
                hasher.update(data);
                hasher.finalize().to_vec()
            },
            HashAlgorithm::Sha3_512 => {
                let mut hasher = Sha3_512::new();
                hasher.update(data);
                hasher.finalize().to_vec()
            },
            HashAlgorithm::Blake3 => {
                blake3::hash(data).as_bytes().to_vec()
            },
        }
    }

    /// Compute HMAC
    pub fn hmac(&self, key: &[u8], data: &[u8]) -> SecurityResult<Vec<u8>> {
        let mut mac = Hmac::<Sha3_256>::new_from_slice(key)
            .map_err(|e| SecurityError::Cryptographic(format!("HMAC key error: {}", e)))?;
        
        mac.update(data);
        Ok(mac.finalize().into_bytes().to_vec())
    }

    /// Generate cryptographically secure random bytes
    pub fn random_bytes(&self, len: usize) -> SecurityResult<Vec<u8>> {
        let mut bytes = vec![0u8; len];
        ring::rand::SecureRandom::fill(&self.rng, &mut bytes)
            .map_err(|e| SecurityError::Cryptographic(format!("Random generation failed: {:?}", e)))?;
        Ok(bytes)
    }

    /// Generate salt for key derivation
    pub fn generate_salt(&self) -> SecurityResult<Vec<u8>> {
        self.random_bytes(self.config.salt_size)
    }

    /// Get public key for sharing
    pub fn public_key(&self) -> [u8; 32] {
        self.signature_keypair.verifying_key_bytes()
    }

    /// Derive key from password
    pub fn derive_key(&self, password: &[u8], salt: &[u8]) -> SymmetricKey {
        SymmetricKey::from_password(password, salt, self.config.kdf_iterations)
    }
}

/// Encrypted data structure
#[derive(Debug, Clone)]
pub struct EncryptedData {
    pub ciphertext: Vec<u8>,
    pub nonce: [u8; 12],
    pub associated_data: Option<Vec<u8>>,
}

impl EncryptedData {
    /// Serialize to bytes for transmission
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        // Length-prefixed format: [nonce_len][nonce][aad_len][aad][ciphertext_len][ciphertext]
        bytes.extend_from_slice(&12u32.to_le_bytes()); // nonce length
        bytes.extend_from_slice(&self.nonce);
        
        if let Some(aad) = &self.associated_data {
            bytes.extend_from_slice(&(aad.len() as u32).to_le_bytes());
            bytes.extend_from_slice(aad);
        } else {
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }
        
        bytes.extend_from_slice(&(self.ciphertext.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.ciphertext);
        
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> SecurityResult<Self> {
        if bytes.len() < 4 {
            return Err(SecurityError::Cryptographic("Invalid encrypted data format".to_string()));
        }

        let mut offset = 0;
        
        // Read nonce
        let nonce_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        offset += 4;
        
        if bytes.len() < offset + nonce_len {
            return Err(SecurityError::Cryptographic("Invalid nonce in encrypted data".to_string()));
        }
        
        let mut nonce = [0u8; 12];
        nonce.copy_from_slice(&bytes[offset..offset + nonce_len]);
        offset += nonce_len;
        
        // Read AAD
        if bytes.len() < offset + 4 {
            return Err(SecurityError::Cryptographic("Invalid AAD length in encrypted data".to_string()));
        }
        
        let aad_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        let associated_data = if aad_len > 0 {
            if bytes.len() < offset + aad_len {
                return Err(SecurityError::Cryptographic("Invalid AAD in encrypted data".to_string()));
            }
            Some(bytes[offset..offset + aad_len].to_vec())
        } else {
            None
        };
        offset += aad_len;
        
        // Read ciphertext
        if bytes.len() < offset + 4 {
            return Err(SecurityError::Cryptographic("Invalid ciphertext length in encrypted data".to_string()));
        }
        
        let ciphertext_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        if bytes.len() < offset + ciphertext_len {
            return Err(SecurityError::Cryptographic("Invalid ciphertext in encrypted data".to_string()));
        }
        
        let ciphertext = bytes[offset..offset + ciphertext_len].to_vec();
        
        Ok(Self {
            ciphertext,
            nonce,
            associated_data,
        })
    }
}

/// Secure memory operations following DoD 5220.22-M standard
pub struct SecureMemory;

impl SecureMemory {
    /// Securely overwrite memory using DoD 5220.22-M 3-pass method
    pub fn secure_erase(data: &mut [u8]) {
        // Pass 1: Overwrite with binary zeros
        data.fill(0x00);
        
        // Pass 2: Overwrite with binary ones
        data.fill(0xFF);
        
        // Pass 3: Overwrite with random pattern
        let mut rng = OsRng;
        for byte in data.iter_mut() {
            *byte = rng.next_u32() as u8;
        }
    }

    /// Secure comparison that resists timing attacks
    pub fn secure_compare(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        let mut result = 0u8;
        for (x, y) in a.iter().zip(b.iter()) {
            result |= x ^ y;
        }
        result == 0
    }

    /// Generate secure random salt
    pub fn generate_salt(size: usize) -> Vec<u8> {
        let mut salt = vec![0u8; size];
        OsRng.fill_bytes(&mut salt);
        salt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_key_generation() {
        let key1 = SymmetricKey::generate();
        let key2 = SymmetricKey::generate();
        
        // Keys should be different
        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_signature_keypair() {
        let keypair = SignatureKeyPair::generate();
        let message = b"test message";
        
        let signature = keypair.sign(message);
        assert!(keypair.verify(message, &signature).is_ok());
    }

    #[test]
    fn test_crypto_context_encryption() {
        let config = CryptoConfig::default();
        let ctx = CryptoContext::new(config).unwrap();
        
        let plaintext = b"Hello, biological security!";
        let encrypted = ctx.encrypt(plaintext, None).unwrap();
        let decrypted = ctx.decrypt(&encrypted).unwrap();
        
        assert_eq!(plaintext, decrypted.as_slice());
    }

    #[test]
    fn test_encryption_with_aad() {
        let config = CryptoConfig::default();
        let ctx = CryptoContext::new(config).unwrap();
        
        let plaintext = b"Secret data";
        let aad = b"Public metadata";
        
        let encrypted = ctx.encrypt(plaintext, Some(aad)).unwrap();
        let decrypted = ctx.decrypt(&encrypted).unwrap();
        
        assert_eq!(plaintext, decrypted.as_slice());
    }

    #[test]
    fn test_encrypted_data_serialization() {
        let config = CryptoConfig::default();
        let ctx = CryptoContext::new(config).unwrap();
        
        let plaintext = b"Test data for serialization";
        let encrypted = ctx.encrypt(plaintext, Some(b"metadata")).unwrap();
        
        let bytes = encrypted.to_bytes();
        let deserialized = EncryptedData::from_bytes(&bytes).unwrap();
        
        let decrypted = ctx.decrypt(&deserialized).unwrap();
        assert_eq!(plaintext, decrypted.as_slice());
    }

    #[test]
    fn test_secure_memory_erase() {
        let mut data = vec![0x42u8; 1024];
        SecureMemory::secure_erase(&mut data);
        
        // After secure erase, data should not contain the original pattern
        assert!(!data.iter().all(|&x| x == 0x42));
    }

    #[test]
    fn test_secure_compare() {
        let data1 = b"identical data";
        let data2 = b"identical data";
        let data3 = b"different data";
        
        assert!(SecureMemory::secure_compare(data1, data2));
        assert!(!SecureMemory::secure_compare(data1, data3));
        assert!(!SecureMemory::secure_compare(data1, b"different length"));
    }

    #[test]
    fn test_hash_algorithms() {
        let config = CryptoConfig::default();
        let ctx = CryptoContext::new(config).unwrap();
        
        let data = b"test data for hashing";
        let hash = ctx.hash(data);
        
        // SHA-3-256 should produce 32-byte hash
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_hmac() {
        let config = CryptoConfig::default();
        let ctx = CryptoContext::new(config).unwrap();
        
        let key = b"secret key";
        let data = b"message to authenticate";
        
        let mac1 = ctx.hmac(key, data).unwrap();
        let mac2 = ctx.hmac(key, data).unwrap();
        
        // Same key and data should produce same MAC
        assert_eq!(mac1, mac2);
        
        // Different data should produce different MAC
        let mac3 = ctx.hmac(key, b"different message").unwrap();
        assert_ne!(mac1, mac3);
    }
}