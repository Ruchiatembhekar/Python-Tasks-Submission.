import numpy as np

class MatrixOperationsTool:
    def __init__(self):
        print("🔢 Welcome to Matrix Operations Tool!")
        print("=" * 40)
    
    def get_matrix_input(self, name):
        """Get matrix input from user"""
        print(f"\nEnter {name}:")
        rows = int(input("Number of rows: "))
        cols = int(input("Number of columns: "))
        
        print(f"Enter {rows}x{cols} matrix elements (space-separated rows):")
        matrix = []
        for i in range(rows):
            row = list(map(float, input(f"Row {i+1}: ").split()))
            if len(row) != cols:
                print(f"Error: Expected {cols} elements, got {len(row)}")
                return self.get_matrix_input(name)
            matrix.append(row)
        
        return np.array(matrix)
    
    def display_matrix(self, matrix, title):
        """Display matrix in formatted way"""
        print(f"\n{title}:")
        print("-" * len(title))
        for row in matrix:
            print(" ".join(f"{elem:8.2f}" for elem in row))
    
    def matrix_addition(self, A, B):
        """Add two matrices"""
        try:
            result = A + B
            self.display_matrix(result, "Addition Result (A + B)")
            return result
        except ValueError:
            print("Error: Matrices must have the same dimensions for addition!")
            return None
    
    def matrix_subtraction(self, A, B):
        """Subtract two matrices"""
        try:
            result = A - B
            self.display_matrix(result, "Subtraction Result (A - B)")
            return result
        except ValueError:
            print("Error: Matrices must have the same dimensions for subtraction!")
            return None
    
    def matrix_multiplication(self, A, B):
        """Multiply two matrices"""
        try:
            result = np.dot(A, B)
            self.display_matrix(result, "Multiplication Result (A × B)")
            return result
        except ValueError:
            print("Error: Number of columns in first matrix must equal number of rows in second matrix!")
            return None
    
    def matrix_transpose(self, matrix):
        """Calculate transpose of matrix"""
        result = np.transpose(matrix)
        self.display_matrix(result, "Transpose Result")
        return result
    
    def matrix_determinant(self, matrix):
        """Calculate determinant of matrix"""
        if matrix.shape[0] != matrix.shape[1]:
            print("Error: Determinant can only be calculated for square matrices!")
            return None
        
        det = np.linalg.det(matrix)
        print(f"\nDeterminant: {det:.4f}")
        return det
    
    def run_interactive_session(self):
        """Run interactive matrix operations session"""
        while True:
            print("\n" + "="*50)
            print("MATRIX OPERATIONS MENU")
            print("="*50)
            print("1. Addition")
            print("2. Subtraction") 
            print("3. Multiplication")
            print("4. Transpose")
            print("5. Determinant")
            print("6. Exit")
            
            choice = input("\nSelect operation (1-6): ").strip()
            
            if choice == '6':
                print("Thank you for using Matrix Operations Tool! 👋")
                break
            
            if choice in ['1', '2', '3']:
                # Operations requiring two matrices
                matrix_A = self.get_matrix_input("Matrix A")
                self.display_matrix(matrix_A, "Matrix A")
                
                matrix_B = self.get_matrix_input("Matrix B")
                self.display_matrix(matrix_B, "Matrix B")
                
                if choice == '1':
                    self.matrix_addition(matrix_A, matrix_B)
                elif choice == '2':
                    self.matrix_subtraction(matrix_A, matrix_B)
                elif choice == '3':
                    self.matrix_multiplication(matrix_A, matrix_B)
            
            elif choice in ['4', '5']:
                # Operations requiring one matrix
                matrix = self.get_matrix_input("Matrix")
                self.display_matrix(matrix, "Input Matrix")
                
                if choice == '4':
                    self.matrix_transpose(matrix)
                elif choice == '5':
                    self.matrix_determinant(matrix)
            
            else:
                print("Invalid choice! Please select 1-6.")
            
            input("\nPress Enter to continue...")

# Run the tool
if __name__ == "__main__":
    tool = MatrixOperationsTool()
    tool.run_interactive_session() 
