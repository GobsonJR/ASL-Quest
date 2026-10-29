import { jsPDF } from "jspdf";

export interface UserExportData {
  exported_at?: string;
  profile?: {
    username?: string;
    email?: string;
    role?: string;
    level?: number;
    current_streak?: number;
    total_xp?: number;
    rank?: string;
    [key: string]: unknown;
  };
  progress?: {
    letterStats?: Record<string, { correct: number; attempts: number }>;
    completedLessons?: string[];
    [key: string]: unknown;
  };
  practice_history?: Array<{
    letter?: string;
    correct?: boolean;
    response_time_ms?: number;
    xp_earned?: number;
    created_at?: string;
    [key: string]: unknown;
  }>;
  achievements?: Array<{
    name?: string;
    slug?: string;
    earned_at?: string;
  }>;
  word_progress?: Record<string, { completions?: number; accuracy?: number }>;
  completed_words?: string[];
  [key: string]: unknown;
}

export function generateProgressPdf(data: UserExportData): void {
  const doc = new jsPDF({
    orientation: "portrait",
    unit: "mm",
    format: "a4",
  });

  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 18;
  const contentWidth = pageWidth - margin * 2;
  let y = 20;

  // Colors
  const primaryColor: [number, number, number] = [46, 125, 50]; // Forest green
  const darkTextColor: [number, number, number] = [30, 41, 59];
  const mutedTextColor: [number, number, number] = [100, 116, 139];
  const cardBgColor: [number, number, number] = [248, 250, 252];
  const borderColor: [number, number, number] = [226, 232, 240];

  // --- Header ---
  doc.setFillColor(...primaryColor);
  doc.rect(margin, y, 4, 18, "F");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(20);
  doc.setTextColor(...darkTextColor);
  doc.text("ASL Quest — Learning Progress Report", margin + 8, y + 8);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(...mutedTextColor);
  const userIdentifier = data.profile?.username || data.profile?.email || "Learner";
  const dateStr = data.exported_at ? new Date(data.exported_at).toLocaleDateString(undefined, { dateStyle: "long" }) : new Date().toLocaleDateString();
  doc.text(`Learner: ${userIdentifier}   |   Generated: ${dateStr}`, margin + 8, y + 15);

  y += 26;

  // --- Calculations ---
  const letterStats = data.progress?.letterStats || {};
  const letters = Object.keys(letterStats);
  const masteredLetters = letters.filter((l) => (letterStats[l]?.correct ?? 0) >= 5);
  const learningLetters = letters.filter((l) => {
    const c = letterStats[l]?.correct ?? 0;
    return c > 0 && c < 5;
  });
  const alphabetMasteryPercent = Math.min(100, Math.round((masteredLetters.length / 26) * 100));

  const history = data.practice_history || [];
  const totalSessions = history.length;
  const correctSessions = history.filter((s) => s.correct).length;
  const accuracyPercent = totalSessions > 0 ? Math.round((correctSessions / totalSessions) * 100) : 0;

  const validTimes = history.map((s) => s.response_time_ms).filter((t): t is number => typeof t === "number" && t > 0);
  const avgTimeMs = validTimes.length > 0 ? Math.round(validTimes.reduce((a, b) => a + b, 0) / validTimes.length) : null;
  const avgTimeSec = avgTimeMs !== null ? (avgTimeMs / 1000).toFixed(1) : "—";

  // Active days aggregation (heatmap data synthesis)
  const activeDaysSet = new Set<string>();
  history.forEach((s) => {
    if (s.created_at) {
      activeDaysSet.add(s.created_at.slice(0, 10));
    }
  });
  const totalActiveDays = activeDaysSet.size;

  const level = data.profile?.level ?? 1;
  const streak = data.profile?.current_streak ?? 0;
  const totalXp = data.profile?.total_xp ?? 0;
  const achievements = data.achievements || [];
  const completedWords = data.completed_words || Object.keys(data.word_progress || {});

  // --- Key Metrics Grid (Cards) ---
  const cardWidth = (contentWidth - 6) / 4;
  const cardHeight = 20;
  const metrics = [
    { label: "Level & XP", value: `Lv. ${level} (${totalXp} XP)` },
    { label: "Current Streak", value: `${streak} day${streak === 1 ? "" : "s"}` },
    { label: "A–Z Mastery", value: `${alphabetMasteryPercent}% (${masteredLetters.length}/26)` },
    { label: "Overall Accuracy", value: `${accuracyPercent}% (${correctSessions}/${totalSessions})` },
  ];

  metrics.forEach((m, idx) => {
    const cx = margin + idx * (cardWidth + 2);
    doc.setFillColor(...cardBgColor);
    doc.setDrawColor(...borderColor);
    doc.roundedRect(cx, y, cardWidth, cardHeight, 2, 2, "FD");

    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(...mutedTextColor);
    doc.text(m.label, cx + 3, y + 6);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(...primaryColor);
    doc.text(m.value, cx + 3, y + 14);
  });

  y += cardHeight + 8;

  // --- Executive Summary / Heatmap Narrative Paragraph ---
  doc.setFont("helvetica", "bold");
  doc.setFontSize(12);
  doc.setTextColor(...darkTextColor);
  doc.text("Learning Activity & Progress Overview", margin, y);
  y += 5;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(...darkTextColor);

  const narrative =
    `To date, you have reached Level ${level} with a total of ${totalXp} XP accumulated. ` +
    `Your active practice history reflects ${totalSessions} total hand sign attempts across ${totalActiveDays} active practice day${totalActiveDays === 1 ? "" : "s"}, ` +
    `maintaining a current streak of ${streak} consecutive day${streak === 1 ? "" : "s"}. ` +
    `Across all practice sessions, your cumulative sign recognition accuracy stands at ${accuracyPercent}%, with an average response time of ${avgTimeSec} seconds per sign. ` +
    `In the A–Z Alphabet curriculum, you have achieved full mastery on ${masteredLetters.length} letter${masteredLetters.length === 1 ? "" : "s"} (${alphabetMasteryPercent}% complete)` +
    (learningLetters.length > 0 ? `, with ${learningLetters.length} additional letter${learningLetters.length === 1 ? "" : "s"} actively in progress.` : ".") +
    (completedWords.length > 0 ? ` Additionally, you have completed practice for ${completedWords.length} vocabulary word${completedWords.length === 1 ? "" : "s"}.` : "") +
    (achievements.length > 0 ? ` You have successfully unlocked ${achievements.length} milestone achievement badge${achievements.length === 1 ? "" : "s"}.` : "");

  const splitNarrative = doc.splitTextToSize(narrative, contentWidth);
  doc.text(splitNarrative, margin, y);
  y += splitNarrative.length * 4.8 + 6;

  // --- Alphabet Mastery Breakdown ---
  doc.setFont("helvetica", "bold");
  doc.setFontSize(12);
  doc.setTextColor(...darkTextColor);
  doc.text("Alphabet Sign Progress (A–Z)", margin, y);
  y += 6;

  // 4 columns table
  const colW = contentWidth / 4;
  const alphabet = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));

  doc.setFont("helvetica", "bold");
  doc.setFontSize(8);
  doc.setTextColor(...mutedTextColor);
  doc.text("LETTER & STATUS", margin + 2, y);
  doc.text("LETTER & STATUS", margin + colW + 2, y);
  doc.text("LETTER & STATUS", margin + colW * 2 + 2, y);
  doc.text("LETTER & STATUS", margin + colW * 3 + 2, y);
  y += 3;

  doc.setDrawColor(...borderColor);
  doc.line(margin, y, margin + contentWidth, y);
  y += 4;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8.5);

  const rows = Math.ceil(26 / 4);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < 4; c++) {
      const idx = r + c * rows;
      if (idx >= 26) continue;
      const letter = alphabet[idx];
      const stats = letterStats[letter];
      const correct = stats?.correct ?? 0;
      const pct = Math.min(100, Math.round((correct / 5) * 100));

      const cx = margin + c * colW;
      let statusText = "Not started";
      let statusColor: [number, number, number] = [148, 163, 184];

      if (correct >= 5) {
        statusText = `Mastered (${pct}%)`;
        statusColor = [22, 163, 74]; // Green
      } else if (correct > 0) {
        statusText = `Learning (${pct}%)`;
        statusColor = [217, 119, 6]; // Amber
      }

      doc.setFont("helvetica", "bold");
      doc.setTextColor(...darkTextColor);
      doc.text(letter, cx + 2, y);

      doc.setFont("helvetica", "normal");
      doc.setTextColor(...statusColor);
      doc.text(statusText, cx + 10, y);
    }
    y += 5.2;
  }

  y += 4;

  // --- Achievements & Milestones ---
  if (achievements.length > 0) {
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(...darkTextColor);
    doc.text("Milestones & Achievements Unlocked", margin, y);
    y += 5;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(8.5);
    doc.setTextColor(...darkTextColor);

    const badgeList = achievements.map((a) => {
      const dt = a.earned_at ? ` (${new Date(a.earned_at).toLocaleDateString()})` : "";
      return `• ${a.name || a.slug}${dt}`;
    });

    const splitBadges = doc.splitTextToSize(badgeList.join("    "), contentWidth);
    doc.text(splitBadges, margin, y);
    y += splitBadges.length * 4.5 + 4;
  }

  // --- Footer ---
  const footerY = doc.internal.pageSize.getHeight() - 12;
  doc.setDrawColor(...borderColor);
  doc.line(margin, footerY - 3, margin + contentWidth, footerY - 3);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(...mutedTextColor);
  doc.text("ASL Quest — Official Learning Record & Skill Summary", margin, footerY);
  doc.text(`Page 1 of 1`, margin + contentWidth - 15, footerY);

  // Trigger download
  const filename = `asl-quest-progress-${userIdentifier.replace(/[^a-zA-Z0-9_-]/g, "_")}.pdf`;
  doc.save(filename);
}
