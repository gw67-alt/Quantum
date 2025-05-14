import numpy as np
from multiprocessing import shared_memory
import pickle
import struct

class SharedSequenceManager:
    """
    A class to manage a shared sequence in shared memory that can be accessed
    across different Python scripts and processes.
    """
    def __init__(self, name="preset_guess_sequence", create=True):
        """
        Initialize the shared memory manager.
        
        Args:
            name (str): Name of the shared memory block
            create (bool): Whether to create a new shared memory or attach to existing one
        """
        self.name = name
        self.create = create
        self.shm = None
        
        if create:
            # We'll allocate space for:
            # - 4 bytes for the sequence length (int32)
            # - Enough space for the pickled sequence
            # Let's start with a reasonable size that can be adjusted
            self.initial_size = 1024 * 10  # 10KB should be enough for most sequences
            self.shm = shared_memory.SharedMemory(
                name=self.name, create=True, size=self.initial_size
            )
            # Initialize with empty sequence
            self.set_sequence([])
        else:
            # Attach to existing shared memory
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            except FileNotFoundError:
                raise RuntimeError(f"Shared memory block '{name}' does not exist. Create it first.")
    
    def set_sequence(self, sequence):
        """Set the sequence in shared memory."""
        if not self.shm:
            raise RuntimeError("Shared memory not initialized")
        
        # Pickle the sequence
        pickled_data = pickle.dumps(sequence)
        pickled_size = len(pickled_data)
        
        # Check if we need to resize the shared memory block
        required_size = 4 + pickled_size  # 4 bytes for length + data
        if required_size > self.shm.size:
            # We need to recreate the shared memory with larger size
            self.shm.close()
            self.shm.unlink()
            new_size = max(required_size, self.shm.size * 2)  # Double the size or use required size
            self.shm = shared_memory.SharedMemory(
                name=self.name, create=True, size=new_size
            )
        
        # Write the length as a 4-byte integer
        self.shm.buf[0:4] = struct.pack('I', pickled_size)
        
        # Write the pickled data
        self.shm.buf[4:4+pickled_size] = pickled_data
    
    def get_sequence(self):
        """Get the sequence from shared memory."""
        if not self.shm:
            raise RuntimeError("Shared memory not initialized")
        
        # Read the length
        pickled_size = struct.unpack('I', bytes(self.shm.buf[0:4]))[0]
        
        # Read the pickled data
        pickled_data = bytes(self.shm.buf[4:4+pickled_size])
        
        # Unpickle and return
        return pickle.loads(pickled_data)
    
    def close(self):
        """Close the shared memory."""
        if self.shm:
            self.shm.close()
    
    def unlink(self):
        """Unlink the shared memory (should be called by the creator only)."""
        if self.create and self.shm:
            self.shm.unlink()
    
    def __del__(self):
        """Clean up when the object is garbage collected."""
        self.close()
        if self.create:
            try:
                self.unlink()
            except:
                pass  # Might already be unlinked

# Example usage as a manager script
if __name__ == "__main__":
    # Create a sample preset sequence
    preset_sequence = [1, 2, 3, 4, 5]
    
    # Create the shared memory manager
    manager = SharedSequenceManager(name="preset_guess_sequence", create=True)
    
    try:
        # Set the sequence
        manager.set_sequence(preset_sequence)
        print(f"Sequence set in shared memory: {preset_sequence}")
        print(f"Shared memory block name: {manager.name}")
        print("Keep this script running to maintain the shared memory.")
        print("Press Ctrl+C to exit and clean up.")
        
        # Keep the script running
        while True:
            cmd = input("Enter 'q' to quit, 'g' to get current sequence, 's <seq>' to set new sequence: ")
            if cmd.lower() == 'q':
                break
            elif cmd.lower() == 'g':
                print(f"Current sequence: {manager.get_sequence()}")
            elif cmd.lower().startswith('s '):
                try:
                    # Parse the new sequence from input
                    new_seq = eval(cmd[2:])
                    if isinstance(new_seq, (list, tuple)):
                        manager.set_sequence(new_seq)
                        print(f"Sequence updated to: {new_seq}")
                    else:
                        print("Input must be a list or tuple")
                except Exception as e:
                    print(f"Error setting sequence: {e}")
            else:
                print("Invalid command")
    
    finally:
        # Clean up
        manager.close()
        manager.unlink()
        print("Shared memory cleaned up.")
