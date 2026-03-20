"""
Memory Encryption Service
==========================

Provides encryption at rest for sensitive memory content.
Uses AES-256-GCM authenticated encryption (military-grade).

This service encrypts:
- Memory content before storage
- Memory embeddings (optional, for high-security mode)
- Anchor text and context

Security Features:
- AES-256-GCM: 256-bit key, authenticated encryption with associated data
- Random 96-bit nonce per encryption (never reused)
- HMAC authentication built into GCM mode
- Secure key derivation using PBKDF2 with SHA-256

Author: Resonant Genesis Team
Date: December 29, 2025
Updated: January 2026 - Upgraded to AES-256-GCM
"""

import os
import base64
import hashlib
import logging
import secrets
import json
from typing import Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import cryptography, fallback gracefully
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography package not available - encryption disabled")


# Encryption constants
AES_256_KEY_SIZE = 32  # 256 bits
GCM_NONCE_SIZE = 12    # 96 bits (recommended for GCM)
GCM_TAG_SIZE = 16      # 128 bits authentication tag


@dataclass
class EncryptionConfig:
    """Configuration for memory encryption."""
    enabled: bool = True  # ENABLED BY DEFAULT for production security
    encrypt_content: bool = True
    encrypt_embeddings: bool = False  # Expensive, only for high-security
    encrypt_anchors: bool = True
    key_rotation_days: int = 90
    algorithm: str = "AES-256-GCM"  # Military-grade encryption


class MemoryEncryptionService:
    """
    Encrypts and decrypts memory content at rest.
    
    Uses AES-256-GCM authenticated encryption which provides:
    - 256-bit AES encryption (military-grade)
    - GCM mode with built-in authentication (AEAD)
    - Random 96-bit nonce per encryption
    - 128-bit authentication tag
    
    
    Environment Variables:
    - MEMORY_ENCRYPTION_KEY: Base64-encoded 256-bit key (44 chars) or passphrase
    - MEMORY_ENCRYPTION_ENABLED: "true" to enable (default: true)
    - MEMORY_ENCRYPTION_SALT: Optional salt for key derivation
    """
    
    def __init__(self):
        """Initialize encryption service."""
        # Default to enabled for production security
        env_enabled = os.getenv("MEMORY_ENCRYPTION_ENABLED", "true").lower()
        
        self.config = EncryptionConfig(
            enabled=env_enabled == "true",
            encrypt_content=True,
            encrypt_embeddings=os.getenv("MEMORY_ENCRYPT_EMBEDDINGS", "false").lower() == "true",
            encrypt_anchors=True,
            algorithm="AES-256-GCM",
        )
        
        self._aesgcm: Optional[AESGCM] = None
        self._key_hash: Optional[str] = None
        self._key: Optional[bytes] = None
        
        if self.config.enabled:
            self._initialize_cipher()
    
    def _derive_key_pbkdf2(self, passphrase: str, salt: Optional[bytes] = None) -> bytes:
        """
        Derive a 256-bit key from passphrase using PBKDF2-SHA256.
        
        Args:
            passphrase: User-provided passphrase or master key
            salt: Optional salt (will use default if not provided)
            
        Returns:
            32-byte (256-bit) derived key
        """
        if salt is None:
            # Use environment salt or derive from passphrase
            env_salt = os.getenv("MEMORY_ENCRYPTION_SALT")
            if env_salt:
                salt = env_salt.encode()
            else:
                # Deterministic salt derived from passphrase (for consistency)
                salt = hashlib.sha256(f"resonant_genesis_salt_{passphrase[:8]}".encode()).digest()[:16]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=AES_256_KEY_SIZE,
            salt=salt,
            iterations=100000,  # OWASP recommended minimum
            backend=default_backend()
        )
        return kdf.derive(passphrase.encode())
    
    def _initialize_cipher(self) -> None:
        """Initialize the AES-256-GCM cipher with encryption key."""
        if not CRYPTO_AVAILABLE:
            logger.error("Encryption enabled but cryptography package not available!")
            self.config.enabled = False
            return
        
        encryption_key = os.getenv("MEMORY_ENCRYPTION_KEY")
        
        if not encryption_key:
            # Try to derive from a master key
            master_key = os.getenv("MASTER_ENCRYPTION_KEY") or os.getenv("JWT_SECRET_KEY")
            if master_key:
                # Derive AES-256 key using PBKDF2
                self._key = self._derive_key_pbkdf2(master_key)
                logger.info("Derived AES-256 key from master key using PBKDF2")
            else:
                logger.warning("No encryption key found - generating ephemeral key (NOT for production!)")
                self._key = secrets.token_bytes(AES_256_KEY_SIZE)
        else:
            # Check if it's a base64-encoded key or passphrase
            try:
                decoded = base64.urlsafe_b64decode(encryption_key)
                if len(decoded) == AES_256_KEY_SIZE:
                    self._key = decoded
                else:
                    # Derive key from passphrase
                    self._key = self._derive_key_pbkdf2(encryption_key)
            except Exception:
                # Treat as passphrase
                self._key = self._derive_key_pbkdf2(encryption_key)
        
        try:
            # Initialize AES-256-GCM
            self._aesgcm = AESGCM(self._key)
            self._key_hash = hashlib.sha256(self._key).hexdigest()[:16]
            logger.info(f"✅ AES-256-GCM encryption initialized (key hash: {self._key_hash})")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self.config.enabled = False
    
    @property
    def is_enabled(self) -> bool:
        """Check if encryption is enabled and functional."""
        return self.config.enabled and self._aesgcm is not None
    
    def encrypt_content(self, plaintext: str) -> Tuple[str, bool]:
        """
        Encrypt memory content using AES-256-GCM.
        
        Args:
            plaintext: The content to encrypt
            
        Returns:
            Tuple of (encrypted_content, was_encrypted)
            - If encryption is disabled, returns (plaintext, False)
            - If encryption succeeds, returns (ciphertext, True)
            
        Format: "ENC2:{base64(nonce + ciphertext + tag)}"
        - ENC2 prefix indicates AES-256-GCM (vs ENC for legacy Fernet)
        - nonce: 12 bytes
        - ciphertext: variable length
        - tag: 16 bytes (appended by GCM)
        """
        if not self.is_enabled or not plaintext:
            return plaintext, False
        
        try:
            # Generate random nonce (NEVER reuse with same key)
            nonce = secrets.token_bytes(GCM_NONCE_SIZE)
            
            # Encrypt with AES-256-GCM
            ciphertext = self._aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
            
            # Combine nonce + ciphertext (tag is appended by GCM)
            encrypted_data = nonce + ciphertext
            
            # Base64 encode and add prefix
            encoded = base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
            return f"ENC2:{encoded}", True
            
        except Exception as e:
            logger.error(f"AES-256-GCM encryption failed: {e}")
            return plaintext, False
    
    def decrypt_content(self, ciphertext: str) -> Tuple[str, bool]:
        """
        Decrypt memory content using AES-256-GCM.
        
        Args:
            ciphertext: The encrypted content (or plaintext if not encrypted)
            
        Returns:
            Tuple of (decrypted_content, was_encrypted)
        """
        if not ciphertext:
            return ciphertext, False
        
        # Check for AES-256-GCM format (ENC2:)
        if ciphertext.startswith("ENC2:"):
            return self._decrypt_aes256gcm(ciphertext)
        
        # Not encrypted
        return ciphertext, False
    
    def _decrypt_aes256gcm(self, ciphertext: str) -> Tuple[str, bool]:
        """Decrypt AES-256-GCM encrypted content."""
        if not self.is_enabled:
            logger.warning("Encrypted content found but encryption is disabled!")
            return ciphertext, False
        
        try:
            # Remove prefix and decode
            encoded_data = ciphertext[5:]  # Remove "ENC2:"
            encrypted_data = base64.urlsafe_b64decode(encoded_data)
            
            # Extract nonce and ciphertext
            nonce = encrypted_data[:GCM_NONCE_SIZE]
            ciphertext_with_tag = encrypted_data[GCM_NONCE_SIZE:]
            
            # Decrypt
            plaintext = self._aesgcm.decrypt(nonce, ciphertext_with_tag, None)
            return plaintext.decode('utf-8'), True
            
        except Exception as e:
            logger.error(f"AES-256-GCM decryption failed: {e}")
            return ciphertext, False
    
    def is_encrypted(self, content: str) -> bool:
        """Check if content appears to be encrypted."""
        if not content:
            return False
        return content.startswith("ENC2:")
    
    def encrypt_embedding(self, embedding: list) -> Tuple[str, bool]:
        """
        Encrypt an embedding vector using AES-256-GCM (for high-security mode).
        
        Args:
            embedding: List of floats representing the embedding
            
        Returns:
            Tuple of (encrypted_embedding_str, was_encrypted)
        """
        if not self.is_enabled or not self.config.encrypt_embeddings:
            return None, False
        
        try:
            embedding_str = json.dumps(embedding)
            
            # Generate random nonce
            nonce = secrets.token_bytes(GCM_NONCE_SIZE)
            
            # Encrypt with AES-256-GCM
            ciphertext = self._aesgcm.encrypt(nonce, embedding_str.encode('utf-8'), None)
            
            # Combine and encode
            encrypted_data = nonce + ciphertext
            encoded = base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
            
            return f"EMB2:{encoded}", True
            
        except Exception as e:
            logger.error(f"Embedding encryption failed: {e}")
            return None, False
    
    def decrypt_embedding(self, encrypted_embedding: str) -> Tuple[list, bool]:
        """
        Decrypt an embedding vector using AES-256-GCM.
        
        Args:
            encrypted_embedding: Encrypted embedding string
            
        Returns:
            Tuple of (embedding_list, was_encrypted)
        """
        if not encrypted_embedding:
            return None, False
        
        # AES-256-GCM format
        if encrypted_embedding.startswith("EMB2:"):
            return self._decrypt_embedding_aes256gcm(encrypted_embedding)
        
        return None, False
    
    def _decrypt_embedding_aes256gcm(self, encrypted_embedding: str) -> Tuple[list, bool]:
        """Decrypt AES-256-GCM encrypted embedding."""
        if not self.is_enabled:
            return None, False
        
        try:
            encoded_data = encrypted_embedding[5:]  # Remove "EMB2:"
            encrypted_data = base64.urlsafe_b64decode(encoded_data)
            
            nonce = encrypted_data[:GCM_NONCE_SIZE]
            ciphertext_with_tag = encrypted_data[GCM_NONCE_SIZE:]
            
            plaintext = self._aesgcm.decrypt(nonce, ciphertext_with_tag, None)
            embedding = json.loads(plaintext.decode('utf-8'))
            return embedding, True
            
        except Exception as e:
            logger.error(f"Embedding decryption failed: {e}")
            return None, False
    
    def rotate_key(self, new_key: Union[str, bytes]) -> bool:
        """
        Rotate to a new encryption key.
        
        Note: This only updates the current key. Existing encrypted data
        must be re-encrypted separately using migrate_encrypted_data().
        
        Args:
            new_key: New 256-bit key (bytes) or passphrase (str)
            
        Returns:
            True if rotation succeeded
        """
        if not CRYPTO_AVAILABLE:
            return False
        
        try:
            # Derive or use key directly
            if isinstance(new_key, bytes):
                if len(new_key) != AES_256_KEY_SIZE:
                    raise ValueError(f"Key must be {AES_256_KEY_SIZE} bytes")
                key_bytes = new_key
            else:
                key_bytes = self._derive_key_pbkdf2(new_key)
            
            # Test new key
            new_aesgcm = AESGCM(key_bytes)
            test_data = b"rotation_test"
            nonce = secrets.token_bytes(GCM_NONCE_SIZE)
            encrypted = new_aesgcm.encrypt(nonce, test_data, None)
            decrypted = new_aesgcm.decrypt(nonce, encrypted, None)
            
            if decrypted != test_data:
                raise ValueError("Key rotation test failed")
            
            # Update keys
            self._key = key_bytes
            self._aesgcm = new_aesgcm
            self._key_hash = hashlib.sha256(key_bytes).hexdigest()[:16]
            
            logger.info(f"✅ Key rotation successful (new key hash: {self._key_hash})")
            return True
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return False
    
    def get_status(self) -> dict:
        """Get encryption service status."""
        return {
            "enabled": self.is_enabled,
            "crypto_available": CRYPTO_AVAILABLE,
            "algorithm": self.config.algorithm,
            "key_size_bits": AES_256_KEY_SIZE * 8,  # 256
            "encrypt_content": self.config.encrypt_content,
            "encrypt_embeddings": self.config.encrypt_embeddings,
            "encrypt_anchors": self.config.encrypt_anchors,
            "key_hash": self._key_hash,
            "key_rotation_days": self.config.key_rotation_days,
            "nonce_size_bits": GCM_NONCE_SIZE * 8,  # 96
            "tag_size_bits": GCM_TAG_SIZE * 8,  # 128
        }


# Global instance
memory_encryption = MemoryEncryptionService()


def encrypt_memory_content(content: str) -> str:
    """Convenience function to encrypt memory content."""
    encrypted, _ = memory_encryption.encrypt_content(content)
    return encrypted


def decrypt_memory_content(content: str) -> str:
    """Convenience function to decrypt memory content."""
    decrypted, _ = memory_encryption.decrypt_content(content)
    return decrypted
