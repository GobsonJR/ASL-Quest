from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    login: str = Field(description="Username or email")
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str = "student"
    created_at: datetime


class LetterStatsSchema(BaseModel):
    correct: int = 0
    attempts: int = 0


class DailyChallengeSchema(BaseModel):
    date: str
    progress: int = 0
    target: int = 10
    completed: bool = False
    mistakes: int = 0
    perfectBonusAwarded: bool = False


class ChallengeProgressSchema(BaseModel):
    alphabetCompleted: bool = False
    speedCompleted: bool = False
    speedStartedAt: int | None = None
    speedCount: int = 0
    wordCompleted: bool = False
    wordCount: int = 0


class RecentAchievementSchema(BaseModel):
    id: str
    title: str
    unlockedAt: str


class GameStateSchema(BaseModel):
    xp: int = 0
    streak: int = 0
    longestStreak: int = 0
    lastPracticeDate: str | None = None
    totalCorrect: int = 0
    totalAttempts: int = 0
    sessions: int = 0
    letterStats: dict[str, LetterStatsSchema]
    unlockedBadges: list[str] = []
    dailyChallenge: DailyChallengeSchema
    challenges: ChallengeProgressSchema
    recentAchievements: list[RecentAchievementSchema] = []


class LetterProgressResponse(BaseModel):
    letter: str
    attempts: int
    correct_attempts: int
    mastery: float
    last_practiced: datetime | None


class LetterProgressUpdate(BaseModel):
    attempts: int
    correct_attempts: int
    mastery: float | None = None


class ChallengeItemResponse(BaseModel):
    challenge_type: str
    progress: int
    completed: bool
    completed_at: datetime | None
    meta: dict


class ChallengeProgressUpdate(BaseModel):
    progress: int
    completed: bool = False
    meta: dict | None = None


class AchievementResponse(BaseModel):
    id: str
    name: str
    description: str
    requirement: str
    earned: bool
    earned_at: datetime | None = None


class PracticeSessionCreate(BaseModel):
    letter: str = Field(min_length=1, max_length=1)
    prediction: str | None = Field(default=None, max_length=1)
    correct: bool
    response_time: int | None = None
    xp_earned: int = 0
    challenge_type: str | None = None
    confidence: float | None = None
    reason: str | None = None


class PracticeSessionResponse(BaseModel):
    id: int
    letter: str
    prediction: str | None
    correct: bool
    response_time: int | None
    xp_earned: int = 0
    challenge_type: str | None = None
    created_at: datetime


class MigrationOfferResponse(BaseModel):
    should_offer: bool
    reason: str | None = None


class WordLetterResult(BaseModel):
    letter: str = Field(min_length=1, max_length=1)
    prediction: str | None = Field(default=None, max_length=1)
    correct: bool
    response_time: int | None = None


class WordPracticeSessionCreate(BaseModel):
    word_id: str
    letters: list[WordLetterResult] = []
    completed: bool = False
    duration_ms: int | None = None
    mistakes: int | None = None
    challenge_type: str | None = None
    challenge_finished: bool = False
    resume_index: int = 0
