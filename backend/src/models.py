from sqlalchemy import Column, Integer, String, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    favorites = relationship("FavoriteProduct", back_populates="user", cascade="all, delete-orphan")
    saved_outfits = relationship("SavedOutfit", back_populates="user", cascade="all, delete-orphan")


class FavoriteProduct(Base):
    __tablename__ = "favorites"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    product_id = Column(Integer, nullable=False)

    user = relationship("User", back_populates="favorites")


class SavedOutfit(Base):
    __tablename__ = "saved_outfits"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    query_product_id = Column(Integer, nullable=False)
    recommendation_text = Column(Text, nullable=False)
    occasion_text = Column(Text, nullable=False)
    # Storing the JSON list of product IDs used in the outfit
    recommended_product_ids = Column(JSON, nullable=False)

    user = relationship("User", back_populates="saved_outfits")
