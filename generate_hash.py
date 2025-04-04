from passlib.context import CryptContext

# Create password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create hash for password "testpass123"
password = "testpass123"
hashed_password = pwd_context.hash(password)
print(f"Hashed password: {hashed_password}") 