#!/usr/bin/env python3
from multiprocessing import shared_memory
import pickle
import struct
import time
import random

class SharedSequenceClient:
    """
    A client to access the preset_guess_sequence from shared memory.
    This can be used in any script that needs to read or modify the sequence.
    """
    def __init__(self, name="preset_guess_sequence"):
        """
        Initialize the client.
        
        Args:
            name (str): Name of the shared memory block to connect to
        """
        self.name = name
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
        except FileNotFoundError:
            raise RuntimeError(
                f"Shared memory block '{name}' not found. "
                "Ensure the manager script is running first."
            )
    
    def get_sequence(self):
        """Get the sequence from shared memory."""
        # Read the length
        pickled_size = struct.unpack('I', bytes(self.shm.buf[0:4]))[0]
        
        # Read the pickled data
        pickled_data = bytes(self.shm.buf[4:4+pickled_size])
        
        # Unpickle and return
        return pickle.loads(pickled_data)
    
    def set_sequence(self, sequence):
        """Set the sequence in shared memory."""
        # Pickle the sequence
        pickled_data = pickle.dumps(sequence)
        pickled_size = len(pickled_data)
        
        # Check if we have enough space
        required_size = 4 + pickled_size
        if required_size > self.shm.size:
            raise ValueError(
                f"Sequence too large for shared memory. "
                f"Required: {required_size}, Available: {self.shm.size}"
            )
        
        # Write the length as a 4-byte integer
        self.shm.buf[0:4] = struct.pack('I', pickled_size)
        
        # Write the pickled data
        self.shm.buf[4:4+pickled_size] = pickled_data
    
    def close(self):
        """Close the connection to shared memory."""
        if hasattr(self, 'shm') and self.shm:
            self.shm.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        """Clean up when the object is garbage collected."""
        self.close()


# Example usage in a client script
if __name__ == "__main__":
    try:
        # Connect to the shared memory
        client = SharedSequenceClient()
        
        # Get the current sequence
        sequence = client.get_sequence()
        print(f"Received sequence: {sequence}")
        
        # Modify the sequence (example: add a random number)
        new_value = random.randint(1, 100)
        sequence.append(new_value)
        print(f"Added new value {new_value}, updated sequence: {sequence}")
        
        # Save the modified sequence back to shared memory
        client.set_sequence(sequence)
        print("Sequence updated in shared memory.")
        
        # Clean up
        client.close()
        
    except Exception as e:
        print(f"Error: {e}")
