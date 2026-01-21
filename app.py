# app.py - Web API version for Render
import re
import math
import requests
from bs4 import BeautifulSoup
from collections import namedtuple
from datetime import datetime
import statistics
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from flask import Flask, request, jsonify, render_template_string
import os

Match = namedtuple('Match', ['date', 'home_team', 'away_team', 'score', 'result', 'goals_scored', 'goals_conceded', 'venue', 'competition', 'opponent_position'])

@dataclass
class H2HMatch:
    """Head-to-head match data"""
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    competition: str

@dataclass
class H2HMetrics:
    """Head-to-head metrics"""
    total_matches: int = 0
    home_wins: int = 0
    away_wins: int = 0
    draws: int = 0
    home_goals_for: float = 0.0
    home_goals_against: float = 0.0
    away_goals_for: float = 0.0
    away_goals_against: float = 0.0
    recent_results: List[str] = None
    draw_rate: float = 0.0
    home_win_rate: float = 0.0
    away_win_rate: float = 0.0
    
    def __post_init__(self):
        if self.recent_results is None:
            self.recent_results = []

@dataclass
class TeamMetrics:
    """Comprehensive team performance metrics computed from match data"""
    team_name: str
    
    # Core metrics computed from last 6 matches
    form_rating: float  # 0-1 scale (points per match normalized)
    goal_scoring_rate: float  # average goals scored per match
    goal_conceding_rate: float  # average goals conceded per match
    clean_sheet_rate: float  # % of matches with clean sheet
    
    # Performance against quality opponents
    top6_opponents_faced: int  # number of top6 opponents in last 6
    performance_vs_top6: float  # 0-1 scale (points per match vs top6)
    top6_performance_score: float  # 0-10 score for performance against top6 teams
    avg_opponent_position: float  # average position of opponents faced
    
    # Match-specific indicators
    scoring_first_rate: float  # % of matches where team scored first
    shots_efficiency: float  # estimated goals per shot (based on scoring)
    possession_dominance: float  # estimated based on results
    home_advantage: float  # home vs away performance difference (calculated from recent matches)
    
    # League position
    league_position: Optional[int]
    
    # Recent momentum
    momentum: float  # -1 to 1 scale (last 3 vs previous 3 matches)
    
    # Goal trends
    goal_trend: float  # -1 to 1 scale (scoring trend)
    defense_trend: float  # -1 to 1 scale (defensive trend)
    
    # Head-to-head data (added)
    h2h_metrics: Optional[H2HMetrics] = None
    
    def __post_init__(self):
        if self.h2h_metrics is None:
            self.h2h_metrics = H2HMetrics()

@dataclass
class MatchFactors:
    """Weighted factors for match prediction"""
    form_weight: float = 0.20
    attack_vs_defense_weight: float = 0.25
    opponent_quality_weight: float = 0.15  # Reduced from 0.20
    home_advantage_weight: float = 0.10  # Reduced from 0.15
    momentum_weight: float = 0.10
    consistency_weight: float = 0.10
    h2h_weight: float = 0.10  # Added H2H weight
    top6_performance_weight: float = 0.05  # Added weight for top6 performance

@dataclass
class CoreAnalysis:
    """Analysis of match determinants based on actual data"""
    form_difference: float = 0.0
    goal_scoring_difference: float = 0.0
    goal_conceding_difference: float = 0.0
    opponent_quality_difference: float = 0.0
    home_advantage_strength: float = 0.0
    momentum_difference: float = 0.0
    top6_performance_difference: float = 0.0  # Added top6 performance difference
    h2h_dominance: float = 0.0  # Added H2H dominance
    key_factors: List[str] = None
    
    def __post_init__(self):
        if self.key_factors is None:
            self.key_factors = []

@dataclass
class MatchSummary:
    """Summary of match characteristics for decision making"""
    is_close_contest: bool = False
    expected_goals_total: float = 0.0
    bts_likely: bool = False
    high_scoring_potential: bool = False
    low_scoring_potential: bool = False
    home_dominant_h2h: bool = False
    away_dominant_h2h: bool = False
    draw_tendency_h2h: bool = False
    characteristics: List[str] = None
    
    def __post_init__(self):
        if self.characteristics is None:
            self.characteristics = []

@dataclass
class DecisionEngine:
    """Engine to make betting decisions based on computed metrics"""
    
    @staticmethod
    def compute_metrics_from_matches(team_name: str, matches: List[Match], 
                                    is_home_team: bool, league_position: Optional[int],
                                    h2h_metrics: Optional[H2HMetrics] = None) -> TeamMetrics:
        """Compute all metrics from actual match data"""
        
        if not matches:
            return TeamMetrics(
                team_name=team_name,
                form_rating=0.5,
                goal_scoring_rate=1.0,
                goal_conceding_rate=1.0,
                clean_sheet_rate=0.3,
                top6_opponents_faced=0,
                performance_vs_top6=0.5,
                top6_performance_score=5.0,
                avg_opponent_position=10.0,
                scoring_first_rate=0.5,
                shots_efficiency=0.15,
                possession_dominance=0.5,
                home_advantage=0.0,  # Default to 0, not 0.6 or 0.4
                league_position=league_position,
                momentum=0.0,
                goal_trend=0.0,
                defense_trend=0.0,
                h2h_metrics=h2h_metrics or H2HMetrics()
            )
        
        # Calculate basic statistics
        total_matches = len(matches)
        
        # Points calculation
        points = 0
        for match in matches:
            if match.result == 'W':
                points += 3
            elif match.result == 'D':
                points += 1
        
        form_rating = points / (total_matches * 3)  # Normalize to 0-1
        
        # Goals statistics
        goals_scored = [m.goals_scored for m in matches]
        goals_conceded = [m.goals_conceded for m in matches]
        
        goal_scoring_rate = sum(goals_scored) / total_matches
        goal_conceding_rate = sum(goals_conceded) / total_matches
        
        # Clean sheet rate
        clean_sheets = sum(1 for m in matches if m.goals_conceded == 0)
        clean_sheet_rate = clean_sheets / total_matches
        
        # Performance against quality opponents (TOP 6)
        top6_opponents = 0
        points_vs_top6 = 0
        matches_vs_top6 = 0
        opponent_positions = []
        
        for match in matches:
            if match.opponent_position:
                opponent_positions.append(match.opponent_position)
                
                if match.opponent_position <= 6:
                    top6_opponents += 1
                    matches_vs_top6 += 1
                    if match.result == 'W':
                        points_vs_top6 += 3
                    elif match.result == 'D':
                        points_vs_top6 += 1
        
        avg_opponent_position = statistics.mean(opponent_positions) if opponent_positions else 10.0
        performance_vs_top6 = points_vs_top6 / (max(1, matches_vs_top6) * 3)
        
        # Calculate top6 performance score (0-10 scale)
        if matches_vs_top6 > 0:
            # Base score from points per match
            base_score = (points_vs_top6 / matches_vs_top6) / 3 * 10
            
            # Adjust based on goal difference against top6
            goals_scored_vs_top6 = sum(m.goals_scored for m in matches if m.opponent_position and m.opponent_position <= 6)
            goals_conceded_vs_top6 = sum(m.goals_conceded for m in matches if m.opponent_position and m.opponent_position <= 6)
            
            if matches_vs_top6 > 0:
                goal_diff_per_match = (goals_scored_vs_top6 - goals_conceded_vs_top6) / matches_vs_top6
                goal_diff_adjustment = goal_diff_per_match * 0.5  # Each goal difference adds 0.5 to score
                top6_performance_score = max(0, min(10, base_score + goal_diff_adjustment))
            else:
                top6_performance_score = 5.0
        else:
            top6_performance_score = 5.0  # Default score if no top6 opponents faced
        
        # Scoring first rate (estimate based on match results)
        scored_first = 0
        for match in matches:
            # Team scored first if they scored and either kept clean sheet or scored more than conceded
            if match.goals_scored > 0 and (match.goals_conceded == 0 or match.goals_scored > match.goals_conceded):
                scored_first += 1
        
        scoring_first_rate = scored_first / total_matches
        
        # Shots efficiency (estimate based on scoring rate)
        # Assume average 5 shots per goal for efficient teams, 10 for inefficient
        if goal_scoring_rate > 2.0:
            shots_efficiency = 0.25  # Very efficient
        elif goal_scoring_rate > 1.5:
            shots_efficiency = 0.20
        elif goal_scoring_rate > 1.0:
            shots_efficiency = 0.15
        elif goal_scoring_rate > 0.5:
            shots_efficiency = 0.10
        else:
            shots_efficiency = 0.05
        
        # Possession dominance (estimate based on results and goal difference)
        if goal_scoring_rate - goal_conceding_rate > 1.0:
            possession_dominance = 0.7  # Dominant team
        elif goal_scoring_rate - goal_conceding_rate > 0.5:
            possession_dominance = 0.6
        elif goal_scoring_rate > goal_conceding_rate:
            possession_dominance = 0.55
        elif goal_scoring_rate < goal_conceding_rate:
            possession_dominance = 0.45
        else:
            possession_dominance = 0.5
        
        # Home advantage calculation - ONLY from recent matches, not default
        home_matches = [m for m in matches if m.venue == 'H']
        away_matches = [m for m in matches if m.venue == 'A']
        
        home_advantage = 0.0  # Start at 0 (neutral)
        
        if home_matches and away_matches:
            home_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in home_matches)
            away_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in away_matches)
            
            max_home_points = len(home_matches) * 3
            max_away_points = len(away_matches) * 3
            
            home_performance = home_points / max_home_points if max_home_points > 0 else 0
            away_performance = away_points / max_away_points if max_away_points > 0 else 0
            
            # Calculate advantage as difference from neutral (0.5)
            home_advantage = home_performance - away_performance  # Range: -1 to +1
        
        # Momentum calculation (last 3 vs previous 3)
        if len(matches) >= 6:
            last_3 = matches[:3]
            previous_3 = matches[3:6]
            
            last_3_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in last_3)
            previous_3_points = sum(3 if m.result == 'W' else 1 if m.result == 'D' else 0 for m in previous_3)
            
            max_points = 9  # 3 matches * 3 points
            momentum = (last_3_points - previous_3_points) / max_points
        else:
            momentum = 0.0
        
        # Goal trends (last 3 vs previous 3)
        if len(matches) >= 6:
            last_3_goals = sum(m.goals_scored for m in matches[:3])
            previous_3_goals = sum(m.goals_scored for m in matches[3:6])
            
            goal_trend = (last_3_goals - previous_3_goals) / 3.0  # Average per match
            goal_trend = max(-1.0, min(1.0, goal_trend / 2.0))  # Normalize to -1 to 1
            
            last_3_conceded = sum(m.goals_conceded for m in matches[:3])
            previous_3_conceded = sum(m.goals_conceded for m in matches[3:6])
            
            defense_trend = (previous_3_conceded - last_3_conceded) / 3.0  # Negative trend = improvement
            defense_trend = max(-1.0, min(1.0, defense_trend / 2.0))  # Normalize to -1 to 1
        else:
            goal_trend = 0.0
            defense_trend = 0.0
        
        return TeamMetrics(
            team_name=team_name,
            form_rating=form_rating,
            goal_scoring_rate=goal_scoring_rate,
            goal_conceding_rate=goal_conceding_rate,
            clean_sheet_rate=clean_sheet_rate,
            top6_opponents_faced=top6_opponents,
            performance_vs_top6=performance_vs_top6,
            top6_performance_score=top6_performance_score,
            avg_opponent_position=avg_opponent_position,
            scoring_first_rate=scoring_first_rate,
            shots_efficiency=shots_efficiency,
            possession_dominance=possession_dominance,
            home_advantage=home_advantage,
            league_position=league_position,
            momentum=momentum,
            goal_trend=goal_trend,
            defense_trend=defense_trend,
            h2h_metrics=h2h_metrics or H2HMetrics()
        )
    
    @staticmethod
    def analyze_core_factors(home_metrics: TeamMetrics, away_metrics: TeamMetrics) -> CoreAnalysis:
        """Analyze core match determinants based on computed metrics"""
        
        analysis = CoreAnalysis()
        key_factors = []
        
        # 1. Form difference
        analysis.form_difference = home_metrics.form_rating - away_metrics.form_rating
        if abs(analysis.form_difference) > 0.3:
            key_factors.append(f"Significant form difference ({analysis.form_difference:+.2f})")
        elif abs(analysis.form_difference) > 0.15:
            key_factors.append(f"Moderate form difference ({analysis.form_difference:+.2f})")
        
        # 2. Goal scoring difference
        analysis.goal_scoring_difference = home_metrics.goal_scoring_rate - away_metrics.goal_scoring_rate
        if analysis.goal_scoring_difference > 0.5:
            key_factors.append(f"{home_metrics.team_name[:10]} scores significantly more")
        elif analysis.goal_scoring_difference < -0.5:
            key_factors.append(f"{away_metrics.team_name[:10]} scores significantly more")
        
        # 3. Goal conceding difference
        analysis.goal_conceding_difference = home_metrics.goal_conceding_rate - away_metrics.goal_conceding_rate
        if analysis.goal_conceding_difference > 0.5:
            key_factors.append(f"{home_metrics.team_name[:10]} concedes significantly more")
        elif analysis.goal_conceding_difference < -0.5:
            key_factors.append(f"{away_metrics.team_name[:10]} concedes significantly more")
        
        # 4. Opponent quality difference
        # Lower avg position = better opponents faced
        home_opponent_strength = 20 - home_metrics.avg_opponent_position if home_metrics.avg_opponent_position else 10
        away_opponent_strength = 20 - away_metrics.avg_opponent_position if away_metrics.avg_opponent_position else 10
        analysis.opponent_quality_difference = home_opponent_strength - away_opponent_strength
        
        if home_metrics.top6_opponents_faced > away_metrics.top6_opponents_faced:
            key_factors.append(f"{home_metrics.team_name[:10]} faced more top6 opponents")
        elif away_metrics.top6_opponents_faced > home_metrics.top6_opponents_faced:
            key_factors.append(f"{away_metrics.team_name[:10]} faced more top6 opponents")
        
        # 5. Home advantage (from recent form, not H2H)
        analysis.home_advantage_strength = home_metrics.home_advantage
        if home_metrics.home_advantage > 0.3:
            key_factors.append(f"Recent home form advantage for {home_metrics.team_name[:10]}")
        elif home_metrics.home_advantage < -0.3:
            key_factors.append(f"Recent poor home form for {home_metrics.team_name[:10]}")
        
        # 6. Momentum difference
        analysis.momentum_difference = home_metrics.momentum - away_metrics.momentum
        if abs(analysis.momentum_difference) > 0.3:
            if analysis.momentum_difference > 0:
                key_factors.append(f"{home_metrics.team_name[:10]} has positive momentum")
            else:
                key_factors.append(f"{away_metrics.team_name[:10]} has positive momentum")
        
        # 7. Top6 Performance difference
        analysis.top6_performance_difference = home_metrics.top6_performance_score - away_metrics.top6_performance_score
        if abs(analysis.top6_performance_difference) > 2.0:
            if analysis.top6_performance_difference > 0:
                key_factors.append(f"{home_metrics.team_name[:10]} stronger against top teams")
            else:
                key_factors.append(f"{away_metrics.team_name[:10]} stronger against top teams")
        
        # 8. H2H dominance (if available)
        if home_metrics.h2h_metrics.total_matches >= 3:
            h2h_home_win_rate = home_metrics.h2h_metrics.home_win_rate
            h2h_away_win_rate = home_metrics.h2h_metrics.away_win_rate
            
            # Calculate dominance (positive favors home, negative favors away)
            analysis.h2h_dominance = h2h_home_win_rate - h2h_away_win_rate
            
            if analysis.h2h_dominance > 0.3:
                key_factors.append(f"H2H dominance: {home_metrics.team_name[:10]} leads {home_metrics.h2h_metrics.home_wins}-{home_metrics.h2h_metrics.draws}-{home_metrics.h2h_metrics.away_wins}")
            elif analysis.h2h_dominance < -0.3:
                key_factors.append(f"H2H dominance: {away_metrics.team_name[:10]} leads {home_metrics.h2h_metrics.away_wins}-{home_metrics.h2h_metrics.draws}-{home_metrics.h2h_metrics.home_wins}")
            elif home_metrics.h2h_metrics.draw_rate > 0.4:
                key_factors.append(f"H2H draw tendency: {home_metrics.h2h_metrics.draws}/{home_metrics.h2h_metrics.total_matches} draws")
        
        analysis.key_factors = key_factors
        return analysis
    
    @staticmethod
    def analyze_match_summary(probabilities: Dict, goal_markets: Dict,
                             home_metrics: TeamMetrics, away_metrics: TeamMetrics,
                             core_analysis: CoreAnalysis) -> MatchSummary:
        """Analyze match characteristics for decision making"""
        
        summary = MatchSummary()
        
        # Determine if it's a close contest
        prob_spread = max(probabilities.values()) - min(probabilities.values())
        summary.is_close_contest = prob_spread < 15  # Less than 15% spread
        
        # Expected goals analysis
        summary.expected_goals_total = goal_markets['total_expected_goals']
        summary.high_scoring_potential = goal_markets['over_25'] > 60
        summary.low_scoring_potential = goal_markets['over_25'] < 40
        summary.bts_likely = goal_markets['bts'] > 60
        
        # H2H patterns
        if home_metrics.h2h_metrics.total_matches >= 3:
            summary.home_dominant_h2h = home_metrics.h2h_metrics.home_win_rate > 0.6
            summary.away_dominant_h2h = home_metrics.h2h_metrics.away_win_rate > 0.6
            summary.draw_tendency_h2h = home_metrics.h2h_metrics.draw_rate > 0.4
        
        # Build characteristics list
        characteristics = []
        
        if summary.is_close_contest:
            characteristics.append("Close contest with small probability spread")
        
        if core_analysis.form_difference > 0.2:
            characteristics.append(f"{home_metrics.team_name[:15]} has better recent form")
        elif core_analysis.form_difference < -0.2:
            characteristics.append(f"{away_metrics.team_name[:15]} has better recent form")
        
        if home_metrics.home_advantage > 0.2:
            characteristics.append(f"{home_metrics.team_name[:15]} shows home advantage")
        elif home_metrics.home_advantage < -0.2:
            characteristics.append(f"{away_metrics.team_name[:15]} shows away form advantage")
        
        if core_analysis.top6_performance_difference > 1.0:
            characteristics.append(f"{home_metrics.team_name[:15]} stronger against top teams")
        elif core_analysis.top6_performance_difference < -1.0:
            characteristics.append(f"{away_metrics.team_name[:15]} stronger against top teams")
        
        if summary.high_scoring_potential:
            characteristics.append("High-scoring potential indicated")
        elif summary.low_scoring_potential:
            characteristics.append("Low-scoring potential indicated")
        
        if summary.bts_likely:
            characteristics.append("Both teams likely to score")
        
        if summary.home_dominant_h2h:
            characteristics.append(f"H2H dominance favors {home_metrics.team_name[:15]}")
        elif summary.away_dominant_h2h:
            characteristics.append(f"H2H dominance favors {away_metrics.team_name[:15]}")
        elif summary.draw_tendency_h2h:
            characteristics.append("Historical tendency for draws")
        
        summary.characteristics = characteristics
        
        return summary
    
    @staticmethod
    def make_decisions(probabilities: Dict, goal_markets: Dict,
                      home_metrics: TeamMetrics, away_metrics: TeamMetrics,
                      core_analysis: CoreAnalysis, match_summary: MatchSummary) -> Dict:
        """Make betting decisions based on computed metrics"""
        
        decisions = {
            'primary_bet': None,
            'secondary_bets': [],
            'confidence': 'LOW',
            'confidence_score': 0,
            'reasoning': [],
            'avoid_bets': [],
            'key_insights': [],
            'match_characteristics': match_summary.characteristics
        }
        
        # CONFIDENCE SCORE CALCULATION (0-10 scale)
        confidence_score = 0
        
        # 1. Strong form difference (+3)
        if abs(core_analysis.form_difference) > 0.3:
            confidence_score += 3
            decisions['key_insights'].append(f"Strong form difference: {core_analysis.form_difference:+.2f}")
        elif abs(core_analysis.form_difference) > 0.15:
            confidence_score += 2
        
        # 2. Significant goal difference (+3)
        if abs(core_analysis.goal_scoring_difference) > 1.0 or abs(core_analysis.goal_conceding_difference) > 1.0:
            confidence_score += 3
            decisions['key_insights'].append(f"Significant goal difference detected")
        
        # 3. Performance against top teams (+2)
        if home_metrics.performance_vs_top6 > 0.7 and away_metrics.performance_vs_top6 < 0.3:
            confidence_score += 2
            decisions['key_insights'].append(f"{home_metrics.team_name[:10]} strong vs top teams")
        elif away_metrics.performance_vs_top6 > 0.7 and home_metrics.performance_vs_top6 < 0.3:
            confidence_score += 2
            decisions['key_insights'].append(f"{away_metrics.team_name[:10]} strong vs top teams")
        
        # 4. Clear momentum (+2)
        if abs(core_analysis.momentum_difference) > 0.4:
            confidence_score += 2
        
        # 5. Strong H2H pattern (+3 if strong, +2 if moderate)
        if home_metrics.h2h_metrics.total_matches >= 3:
            if abs(core_analysis.h2h_dominance) > 0.4:
                confidence_score += 3
                decisions['key_insights'].append("Strong H2H dominance pattern")
            elif abs(core_analysis.h2h_dominance) > 0.2:
                confidence_score += 2
            elif home_metrics.h2h_metrics.draw_rate > 0.5:
                confidence_score += 2
                decisions['key_insights'].append("Strong H2H draw pattern")
        
        # 6. Clear home advantage from recent form (+1)
        if home_metrics.home_advantage > 0.3:
            confidence_score += 1
        
        # 7. Top6 performance difference (+2 if significant)
        if abs(core_analysis.top6_performance_difference) > 3.0:
            confidence_score += 2
            if core_analysis.top6_performance_difference > 0:
                decisions['key_insights'].append(f"{home_metrics.team_name[:10]} excels against top teams")
            else:
                decisions['key_insights'].append(f"{away_metrics.team_name[:10]} excels against top teams")
        
        # Set confidence level based on score
        decisions['confidence_score'] = confidence_score
        if confidence_score >= 8:
            decisions['confidence'] = 'HIGH'
        elif confidence_score >= 5:
            decisions['confidence'] = 'MEDIUM'
        else:
            decisions['confidence'] = 'LOW'
        
        # Determine stake suggestion based on confidence
        stake_map = {
            'HIGH': 'MEDIUM',
            'MEDIUM': 'SMALL',
            'LOW': 'VERY SMALL'
        }
        stake_suggestion = stake_map[decisions['confidence']]
        
        # PRIMARY BET DECISION
        home_prob = probabilities['home_win']
        away_prob = probabilities['away_win']
        draw_prob = probabilities['draw']
        
        # Apply form difference adjustment
        form_adjustment = core_analysis.form_difference * 0.2
        if form_adjustment > 0:
            home_prob = min(85, home_prob * (1 + form_adjustment))
        else:
            away_prob = min(85, away_prob * (1 + form_adjustment))
        
        # Apply home advantage adjustment (only if recent form shows it)
        if home_metrics.home_advantage > 0.2:
            home_advantage_adjustment = home_metrics.home_advantage * 0.2
            home_prob = min(85, home_prob * (1 + home_advantage_adjustment))
        
        # Apply Top6 performance adjustment
        top6_adjustment = core_analysis.top6_performance_difference * 0.02  # 2% per point difference
        if top6_adjustment > 0:
            home_prob = min(85, home_prob * (1 + top6_adjustment))
        else:
            away_prob = min(85, away_prob * (1 - top6_adjustment))
        
        # Apply H2H adjustment if available
        if home_metrics.h2h_metrics.total_matches >= 3:
            h2h_weight = 0.15  # 15% weight for H2H
            
            if core_analysis.h2h_dominance > 0.3:
                # Home team dominates H2H
                home_prob = min(85, home_prob * (1 + h2h_weight))
                decisions['reasoning'].append(f"H2H advantage: {home_metrics.team_name[:10]} leads {home_metrics.h2h_metrics.home_wins}-{home_metrics.h2h_metrics.draws}-{home_metrics.h2h_metrics.away_wins}")
            elif core_analysis.h2h_dominance < -0.3:
                # Away team dominates H2H
                away_prob = min(85, away_prob * (1 + h2h_weight))
                decisions['reasoning'].append(f"H2H advantage: {away_metrics.team_name[:10]} leads {home_metrics.h2h_metrics.away_wins}-{home_metrics.h2h_metrics.draws}-{home_metrics.h2h_metrics.home_wins}")
            elif home_metrics.h2h_metrics.draw_rate > 0.4:
                # Draw pattern in H2H
                draw_prob = min(60, draw_prob * (1 + h2h_weight))
                decisions['reasoning'].append(f"H2H draw tendency: {home_metrics.h2h_metrics.draws}/{home_metrics.h2h_metrics.total_matches} draws")
        
        # Normalize probabilities
        total = home_prob + away_prob + draw_prob
        if total > 0:
            home_prob = home_prob / total * 100
            away_prob = away_prob / total * 100
            draw_prob = draw_prob / total * 100
        
        # Make primary bet decision
        primary_reasoning = []
        
        # Check for draw based on probabilities and match summary
        if (draw_prob > max(home_prob, away_prob) + 5 and draw_prob > 35) or match_summary.draw_tendency_h2h:
            decisions['primary_bet'] = {
                'type': 'MATCH_RESULT',
                'selection': 'DRAW',
                'probability': draw_prob,
                'odds_range': '3.00-3.50',
                'stake_suggestion': stake_suggestion,
                'reasoning': 'Evenly matched teams with draw indicators'
            }
            
            if match_summary.draw_tendency_h2h:
                primary_reasoning.append(f"H2H shows {home_metrics.h2h_metrics.draws}/{home_metrics.h2h_metrics.total_matches} draws")
            if match_summary.is_close_contest:
                primary_reasoning.append("Close contest with small probability spread")
            if core_analysis.form_difference < 0.2:
                primary_reasoning.append("Similar recent form")
        
        elif home_prob > away_prob + 5:
            decisions['primary_bet'] = {
                'type': 'MATCH_RESULT',
                'selection': 'HOME_WIN',
                'probability': home_prob,
                'odds_range': '1.70-2.10' if home_prob > 55 else '2.10-2.50',
                'stake_suggestion': stake_suggestion,
                'reasoning': 'Superior form and/or home advantage'
            }
            
            if core_analysis.form_difference > 0.15:
                primary_reasoning.append(f"Better recent form ({home_metrics.form_rating:.2f} vs {away_metrics.form_rating:.2f})")
            if home_metrics.home_advantage > 0.2:
                primary_reasoning.append(f"Recent home advantage ({home_metrics.home_advantage:+.2f})")
            if match_summary.home_dominant_h2h:
                primary_reasoning.append(f"H2H dominance: {home_metrics.h2h_metrics.home_wins}-{home_metrics.h2h_metrics.draws}-{home_metrics.h2h_metrics.away_wins}")
            if core_analysis.top6_performance_difference > 1.0:
                primary_reasoning.append(f"Stronger against top teams ({home_metrics.top6_performance_score:.1f}/10 vs {away_metrics.top6_performance_score:.1f}/10)")
            if home_metrics.goal_scoring_rate > away_metrics.goal_scoring_rate + 0.5:
                primary_reasoning.append(f"Higher scoring rate ({home_metrics.goal_scoring_rate:.1f} vs {away_metrics.goal_scoring_rate:.1f})")
        
        else:
            decisions['primary_bet'] = {
                'type': 'MATCH_RESULT',
                'selection': 'AWAY_WIN',
                'probability': away_prob,
                'odds_range': '2.20-2.80' if away_prob > 45 else '2.80-3.50',
                'stake_suggestion': stake_suggestion,
                'reasoning': 'Strong away form and/or performance'
            }
            
            if core_analysis.form_difference < -0.15:
                primary_reasoning.append(f"Better recent form ({away_metrics.form_rating:.2f} vs {home_metrics.form_rating:.2f})")
            if match_summary.away_dominant_h2h:
                primary_reasoning.append(f"H2H dominance: {home_metrics.h2h_metrics.away_wins}-{home_metrics.h2h_metrics.draws}-{home_metrics.h2h_metrics.home_wins}")
            if core_analysis.top6_performance_difference < -1.0:
                primary_reasoning.append(f"Stronger against top teams ({away_metrics.top6_performance_score:.1f}/10 vs {home_metrics.top6_performance_score:.1f}/10)")
            if away_metrics.goal_scoring_rate > home_metrics.goal_scoring_rate + 0.5:
                primary_reasoning.append(f"Higher scoring rate ({away_metrics.goal_scoring_rate:.1f} vs {home_metrics.goal_scoring_rate:.1f})")
            if away_metrics.performance_vs_top6 > home_metrics.performance_vs_top6 + 0.3:
                primary_reasoning.append(f"Better performance against top teams")
        
        # Add primary reasoning
        if primary_reasoning:
            decisions['primary_bet']['detailed_reasoning'] = " | ".join(primary_reasoning)
        
        # GOAL MARKET BETS
        if goal_markets['over_25'] > 65:
            decisions['secondary_bets'].append({
                'type': 'GOALS',
                'selection': 'OVER_2.5',
                'probability': goal_markets['over_25'],
                'confidence': 'HIGH' if goal_markets['over_25'] > 75 else 'MEDIUM',
                'reasoning': f"High scoring teams: {home_metrics.goal_scoring_rate:.1f} & {away_metrics.goal_scoring_rate:.1f} avg goals"
            })
        elif goal_markets['over_25'] < 35:
            decisions['secondary_bets'].append({
                'type': 'GOALS',
                'selection': 'UNDER_2.5',
                'probability': 100 - goal_markets['over_25'],
                'confidence': 'HIGH' if goal_markets['over_25'] < 25 else 'MEDIUM',
                'reasoning': f"Low scoring teams: {home_metrics.goal_scoring_rate:.1f} & {away_metrics.goal_scoring_rate:.1f} avg goals"
            })
        
        if goal_markets['bts'] > 65:
            decisions['secondary_bets'].append({
                'type': 'BOTH_TEAMS_SCORE',
                'selection': 'YES',
                'probability': goal_markets['bts'],
                'confidence': 'HIGH' if goal_markets['bts'] > 75 else 'MEDIUM',
                'reasoning': f"Both teams scoring regularly"
            })
        elif goal_markets['bts'] < 35:
            decisions['secondary_bets'].append({
                'type': 'BOTH_TEAMS_SCORE',
                'selection': 'NO',
                'probability': 100 - goal_markets['bts'],
                'confidence': 'HIGH' if goal_markets['bts'] < 25 else 'MEDIUM',
                'reasoning': f"Strong defensive records"
            })
        
        # CLEAN SHEET BETS
        if home_metrics.clean_sheet_rate > 0.5 and away_metrics.goal_scoring_rate < 1.0:
            decisions['secondary_bets'].append({
                'type': 'CLEAN_SHEET',
                'selection': f'{home_metrics.team_name[:10]} YES',
                'probability': 60 + (home_metrics.clean_sheet_rate * 20),
                'confidence': 'MEDIUM',
                'reasoning': f"{home_metrics.team_name[:10]} clean sheet rate: {home_metrics.clean_sheet_rate:.0%}"
            })
        
        if away_metrics.clean_sheet_rate > 0.5 and home_metrics.goal_scoring_rate < 1.0:
            decisions['secondary_bets'].append({
                'type': 'CLEAN_SHEET',
                'selection': f'{away_metrics.team_name[:10]} YES',
                'probability': 60 + (away_metrics.clean_sheet_rate * 20),
                'confidence': 'MEDIUM',
                'reasoning': f"{away_metrics.team_name[:10]} clean sheet rate: {away_metrics.clean_sheet_rate:.0%}"
            })
        
        # RISK WARNINGS
        if home_metrics.form_rating < 0.3:
            decisions['avoid_bets'].append(f"Avoid {home_metrics.team_name[:10]} - poor recent form")
        if away_metrics.form_rating < 0.3:
            decisions['avoid_bets'].append(f"Avoid {away_metrics.team_name[:10]} - poor recent form")
        
        if home_metrics.avg_opponent_position > 15 and home_metrics.form_rating > 0.7:
            decisions['avoid_bets'].append(f"Caution: {home_metrics.team_name[:10]} faced weak opponents")
        if away_metrics.avg_opponent_position > 15 and away_metrics.form_rating > 0.7:
            decisions['avoid_bets'].append(f"Caution: {away_metrics.team_name[:10]} faced weak opponents")
        
        # Add H2H data warning if limited
        if home_metrics.h2h_metrics.total_matches == 0:
            decisions['avoid_bets'].append("No historical H2H data available")
        elif home_metrics.h2h_metrics.total_matches < 3:
            decisions['avoid_bets'].append(f"Limited H2H data ({home_metrics.h2h_metrics.total_matches} matches)")
        
        return decisions

class ForebetAnalyzer:
    """Main analyzer class that extracts and computes all metrics from Forebet data"""
    
    def __init__(self):
        self.factors = MatchFactors()
        self.decision_engine = DecisionEngine()
        self.league_standings = {}
    
    def fetch_page_soup(self, url: str):
        """Fetch page using requests only (works on Render)"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            html = response.text
            
            # If page looks empty, try with different headers
            if len(html) < 1000:
                # Try with simpler headers
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                }
                response = requests.get(url, headers=headers, timeout=30)
                html = response.text
            
            soup = BeautifulSoup(html, "html.parser")
            return soup
            
        except Exception as e:
            print(f"Error fetching page: {e}")
            return None
    
    def clean_team_name(self, name: str):
        """Clean team name for display"""
        if not name or not isinstance(name, str):
            return ""
        
        original = name
        
        if 'showhint' in name.lower():
            match = re.search(r"showhint\('([^']+)'\)", name)
            if match:
                name = match.group(1)
            else:
                name = re.sub(r'showhint\([^)]*\)', '', name, flags=re.IGNORECASE)
        
        name = re.sub(r'javascript:[^)]*\)', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*(?:Prediction|Preview|Stats|H2H|Odds|Forebet|Tips).*$', '', name, flags=re.IGNORECASE)
        
        if '(' in name and ')' in name:
            content = re.search(r'\(([^)]+)\)', name)
            if content:
                content_text = content.group(1)
                if re.search(r'\d', content_text) or any(word in content_text.lower() for word in ['min', 'pen', 'og', 'ht', 'ft', 'agg']):
                    name = re.sub(r'\s*\([^)]*\)', '', name)
        
        name = re.sub(r'^[\'"\']+|[\'"\']+$', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Additional cleaning
        name = re.sub(r'\s*\([^)]*\)', '', name)
        name = re.sub(r'\s+(U\d+|Reserves|B|II|U23|U19)$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+FC\b|\s+CF\b|\s+SC\b|\s+AFC\b', '', name, flags=re.IGNORECASE)
        name = ' '.join(name.split())
        
        if not name or len(name) < 2:
            return original.strip()
        
        return name.strip()
    
    def normalize_name(self, s: str):
        """Normalize team name for comparison"""
        if not s or not isinstance(s, str):
            return ""
        s = s.lower()
        s = re.sub(r'[^\w\s\-\']', '', s)
        s = re.sub(r'\b(jong|u23|u21|ii|2|reserves|youth|academy|prediction|preview|stats|h2h)\b', '', s)
        s = re.sub(r'\s+(fc|cf|sc|afc|cfc|united|city|town|rovers|athletic|club|dept)\b', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    
    def extract_team_names_and_positions(self, soup: BeautifulSoup):
        """Extract team names and their league positions from Forebet page"""
        if not soup:
            return None, None, None, None
        
        home_team, away_team = None, None
        home_position, away_position = None, None
        
        # Look for team names in the page header
        header_selectors = [
            'div.team-name',
            'span.team-name',
            'h1.team-name',
            'div.team_info',
            'div.match-header'
        ]
        
        for selector in header_selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text(strip=True)
                if ' vs ' in text.lower() or ' - ' in text:
                    parts = re.split(r' vs | - ', text, flags=re.IGNORECASE)
                    if len(parts) >= 2:
                        home_team = self.clean_team_name(parts[0])
                        away_team = self.clean_team_name(parts[1])
                        break
        
        # If not found in headers, try the title
        if not home_team or not away_team:
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text()
                patterns = [
                    r'([^\-]+)\s+vs\s+([^\-]+)\s+Prediction',
                    r'([^\-]+)\s+vs\s+([^\-]+)\s+-\s+\d',
                    r'([^\-]+)\s+VS\s+([^\-]+)',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, title_text, re.IGNORECASE)
                    if match:
                        home_team = self.clean_team_name(match.group(1))
                        away_team = self.clean_team_name(match.group(2))
                        break
        
        # Now extract the full standings table
        standings = self.extract_standings_table(soup)
        
        # Get positions for the teams
        if home_team:
            home_position = self.get_position_from_standings(standings, home_team)
        if away_team:
            away_position = self.get_position_from_standings(standings, away_team)
        
        # Update global standings
        for team, pos in standings.items():
            self.league_standings[team] = pos
        
        # Add the match teams if not already in standings
        if home_team and home_team not in self.league_standings and home_position:
            self.league_standings[home_team] = home_position
        if away_team and away_team not in self.league_standings and away_position:
            self.league_standings[away_team] = away_position
        
        return home_team, away_team, home_position, away_position
    
    def extract_standings_table(self, soup: BeautifulSoup):
        """Extract the full standings table from Forebet page"""
        standings = {}
        
        # Try multiple methods to find the standings
        standings_selectors = [
            'table.standing-table',
            'table.stage-standing',
            'table.stats',
            'table#table-standings',
            'table.standings',
            'table[class*="standing"]',
            'table[class*="rank"]',
            'table[class*="table"]',
            'table'
        ]
        
        for selector in standings_selectors:
            try:
                tables = soup.select(selector)
                for table in tables:
                    # Check if this looks like a standings table
                    text = table.get_text()
                    if any(keyword in text.lower() for keyword in ['pts', 'gp', 'w', 'd', 'l', 'gf', 'ga', '+/-', 'position']):
                        rows = table.find_all("tr")
                        for row in rows[1:]:  # Skip header row
                            cols = row.find_all("td")
                            if len(cols) >= 3:
                                # Position is usually in first column
                                pos_text = cols[0].get_text(strip=True)
                                if pos_text.isdigit():
                                    position = int(pos_text)
                                    
                                    # Try to find team name in subsequent columns
                                    team_name = None
                                    for col in cols[1:]:
                                        col_text = col.get_text(strip=True)
                                        if col_text and not col_text.isdigit() and len(col_text) > 2:
                                            team_name = self.clean_team_name(col_text)
                                            if team_name:
                                                break
                                    
                                    if team_name:
                                        standings[team_name] = position
                        
                        if standings:
                            return standings
            except Exception as e:
                continue
        
        # If no table found, try to extract from text patterns
        html_text = str(soup)
        
        # Pattern for standings: position, team name, points, etc.
        patterns = [
            r'(\d+)\s+([A-Za-z\s\-\.]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\+\-]?\d+)',
            r'(\d+)\.\s+([A-Za-z\s\-\.]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\+\-]?\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if len(match) >= 2:
                    position = int(match[0])
                    team_name = self.clean_team_name(match[1])
                    if team_name:
                        standings[team_name] = position
        
        return standings
    
    def get_position_from_standings(self, standings: dict, team_name: str):
        """Get position for a specific team from standings"""
        if not standings or not team_name:
            return None
        
        team_norm = self.normalize_name(team_name)
        
        # Exact match first
        for team, pos in standings.items():
            if self.normalize_name(team) == team_norm:
                return pos
        
        # Partial match
        for team, pos in standings.items():
            team_norm_db = self.normalize_name(team)
            if team_norm in team_norm_db or team_norm_db in team_norm:
                return pos
        
        return None
    
    def extract_head_to_head_matches(self, soup: BeautifulSoup, home_team: str, away_team: str) -> List[H2HMatch]:
        """Extract head-to-head matches from Forebet page"""
        h2h_matches = []
        
        home_norm = self.normalize_name(home_team)
        away_norm = self.normalize_name(away_team)

        # Find containers that might contain H2H data
        containers = []
        for tag in soup.find_all(['div', 'section', 'article', 'table']):
            txt = (tag.get_text() or "").lower()
            if 'head to head' in txt or 'h2h' in txt or 'head' in txt:
                containers.append(tag)

        if not containers:
            containers = [soup]  # fallback - search whole page

        # Patterns for extracting H2H matches
        patterns = [
            r'(\d{2}/\d{2})\s+(\d{4})\s+\[\s*([^]]+?)\s*\]\([^)]+\)\s+(\d+)\s*-\s*(\d+)\s*(?:\([^)]*\))?\s+\[\s*([^]]+?)\s*\]\([^)]+\)\s+([A-Za-z0-9]{2,5})',
            r'(\d{2}/\d{2})\s{2,20}(\d{4})\s{2,30}([^0-9]+?)\s{2,}(\d+)\s*[-–]\s*(\d+)\s*(?:\([^)]+\))?\s{2,}([^0-9]+?)(?:\s+([A-Za-z0-9]{2,6}))?(?=\s{2,}|$|<)',
            r'(\d{2}/\d{2}/\d{4})\s+([^0-9]+?)\s+(\d+)\s*[-–]\s*(\d+)\s*(?:\([^)]+\))?\s+([^0-9]+?)(?:\s+([A-Za-z0-9]{2,5}))?(?=\s|$|<)',
            r'(\d{2}/\d{2})\s*(\d{4})\s+([^0-9]+?)\s+(\d+)\s*-\s*(\d+)\s+([^0-9]+?)(?:\s+([A-Z]{2,5}))?',
            r'(\d{2}/\d{2}/\d{4})\s+([^0-9]+?)\s+(\d+)\s*[-–]\s*(\d+)\s+([^0-9]+?)\s+([A-Za-z]{2,5})',
        ]

        for container in containers:
            text = container.get_text(separator='  ', strip=True)
            
            for pattern in patterns:
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    try:
                        if len(m.groups()) >= 5:
                            date_parts = m.group(1)
                            year_part = m.group(2) if len(m.groups()) > 5 else ""
                            
                            if len(year_part) == 2:
                                year = "20" + year_part
                            elif len(year_part) == 4:
                                year = year_part
                            else:
                                if '/' in date_parts and len(date_parts.split('/')) == 3:
                                    day, month, year = date_parts.split('/')
                                    if len(year) == 2:
                                        year = "20" + year
                                else:
                                    year = str(datetime.now().year)
                            
                            # Construct date string
                            if '/' in date_parts and len(date_parts.split('/')) == 2:
                                date_str = f"{date_parts}/{year}"
                            else:
                                date_str = date_parts

                            team1 = self.clean_team_name(m.group(3))
                            score1 = int(m.group(4))
                            score2 = int(m.group(5))
                            team2 = self.clean_team_name(m.group(6))
                            comp = m.group(7).strip() if len(m.groups()) > 6 and m.group(7) else "Unknown"

                            t1n = self.normalize_name(team1)
                            t2n = self.normalize_name(team2)

                            # Check if it's our match-up
                            if (home_norm in t1n and away_norm in t2n) or \
                               (home_norm in t2n and away_norm in t1n) or \
                               (any(word in t1n for word in home_norm.split() if len(word) > 2) and 
                                any(word in t2n for word in away_norm.split() if len(word) > 2)):

                                h2h_matches.append(H2HMatch(
                                    date=date_str,
                                    home_team=team1,
                                    away_team=team2,
                                    home_goals=score1,
                                    away_goals=score2,
                                    competition=comp
                                ))

                    except (ValueError, IndexError, AttributeError):
                        continue

        # Deduplicate & sort
        seen = set()
        unique = []
        for m in h2h_matches:
            key = (m.date, m.home_team, m.away_team, m.home_goals, m.away_goals)
            if key not in seen:
                seen.add(key)
                unique.append(m)

        unique.sort(key=lambda x: self.parse_date_to_sortable(x.date), reverse=True)
        
        return unique[:12]   # Return up to 12 most recent matches
    
    def calculate_h2h_metrics(self, h2h_matches: List[H2HMatch], home_team: str, away_team: str) -> H2HMetrics:
        """Calculate H2H metrics from match data"""
        
        metrics = H2HMetrics()
        
        if not h2h_matches:
            return metrics
        
        home_norm = self.normalize_name(home_team)
        away_norm = self.normalize_name(away_team)
        
        for match in h2h_matches:
            home_norm_match = self.normalize_name(match.home_team)
            away_norm_match = self.normalize_name(match.away_team)
            
            metrics.total_matches += 1
            
            # Determine winner
            if match.home_goals > match.away_goals:
                if home_norm in home_norm_match or any(word in home_norm_match for word in home_norm.split() if len(word) > 2):
                    metrics.home_wins += 1
                    metrics.recent_results.append('H')
                else:
                    metrics.away_wins += 1
                    metrics.recent_results.append('A')
            elif match.home_goals < match.away_goals:
                if away_norm in away_norm_match or any(word in away_norm_match for word in away_norm.split() if len(word) > 2):
                    metrics.away_wins += 1
                    metrics.recent_results.append('A')
                else:
                    metrics.home_wins += 1
                    metrics.recent_results.append('H')
            else:
                metrics.draws += 1
                metrics.recent_results.append('D')
            
            # Track goals
            if home_norm in home_norm_match or any(word in home_norm_match for word in home_norm.split() if len(word) > 2):
                metrics.home_goals_for += match.home_goals
                metrics.home_goals_against += match.away_goals
                metrics.away_goals_for += match.away_goals
                metrics.away_goals_against += match.home_goals
            else:
                metrics.home_goals_for += match.away_goals
                metrics.home_goals_against += match.home_goals
                metrics.away_goals_for += match.home_goals
                metrics.away_goals_against += match.away_goals
        
        # Calculate rates
        if metrics.total_matches > 0:
            metrics.draw_rate = metrics.draws / metrics.total_matches
            metrics.home_win_rate = metrics.home_wins / metrics.total_matches
            metrics.away_win_rate = metrics.away_wins / metrics.total_matches
            
            # Average goals
            if metrics.home_wins + metrics.draws > 0:
                metrics.home_goals_for = metrics.home_goals_for / metrics.total_matches
                metrics.home_goals_against = metrics.home_goals_against / metrics.total_matches
            
            if metrics.away_wins + metrics.draws > 0:
                metrics.away_goals_for = metrics.away_goals_for / metrics.total_matches
                metrics.away_goals_against = metrics.away_goals_against / metrics.total_matches
        
        return metrics
    
    def get_last_six_matches(self, html: str, team_name: str, competition_name: str = None):
        """Extract recent matches using improved regex patterns"""
        if not html or not team_name:
            return []
        
        matches = []
        clean_team = self.clean_team_name(team_name)
        team_normalized = self.normalize_name(clean_team)
        
        if not team_normalized:
            return []
        
        # Improved regex patterns for Forebet match format
        patterns = [
            # Date (dd/mm/yyyy) + Home Team + Score + Away Team
            r'(\d{2}/\d{2}/\d{4})\s+([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+)',
            # Date (dd/mm/yy) + Home Team + Score + Away Team
            r'(\d{2}/\d{2}/\d{2})\s+([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+)',
            # Date (dd/mm) + Home Team + Score + Away Team
            r'(\d{2}/\d{2})\s+([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+)',
            # Home Team + Score + Away Team (no date)
            r'([^0-9\-]+?)\s+(\d+)\s*[–\-]\s*(\d+)\s+([^0-9\-]+)',
        ]
        
        all_found_matches = []
        
        for pattern in patterns:
            found_matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
            all_found_matches.extend(found_matches)
        
        # Process matches
        for match_tuple in all_found_matches:
            try:
                if len(match_tuple) == 5:
                    if re.match(r'\d{2}/\d{2}', match_tuple[0]):
                        date, team1, g1, g2, team2 = match_tuple
                    else:
                        team1, g1, g2, team2, competition = match_tuple
                        date = ""
                elif len(match_tuple) == 4:
                    team1, g1, g2, team2 = match_tuple
                    date = ""
                    competition = ""
                else:
                    continue
                
                # Clean team names
                team1_clean = self.clean_team_name(team1.strip())
                team2_clean = self.clean_team_name(team2.strip())
                
                if not team1_clean or not team2_clean:
                    continue
                
                # Convert scores
                try:
                    g1_int = int(g1.strip())
                    g2_int = int(g2.strip())
                except ValueError:
                    continue
                
                # Check if our team is involved
                team1_norm = self.normalize_name(team1_clean)
                team2_norm = self.normalize_name(team2_clean)
                
                opponent_position = None
                
                if team_normalized in team1_norm or team1_norm in team_normalized:
                    result = "W" if g1_int > g2_int else "D" if g1_int == g2_int else "L"
                    venue = 'H'
                    goals_scored = g1_int
                    goals_conceded = g2_int
                    opponent = team2_clean
                    opponent_position = self.get_position_from_standings(self.league_standings, team2_clean)
                    
                    match_obj = Match(
                        date=date.strip() if date else "",
                        home_team=team1_clean,
                        away_team=team2_clean,
                        score=f"{g1_int}-{g2_int}",
                        result=result,
                        goals_scored=goals_scored,
                        goals_conceded=goals_conceded,
                        venue=venue,
                        competition=competition.strip() if 'competition' in locals() else "Unknown",
                        opponent_position=opponent_position
                    )
                    matches.append(match_obj)
                    
                elif team_normalized in team2_norm or team2_norm in team_normalized:
                    result = "W" if g2_int > g1_int else "D" if g2_int == g1_int else "L"
                    venue = 'A'
                    goals_scored = g2_int
                    goals_conceded = g1_int
                    opponent = team1_clean
                    opponent_position = self.get_position_from_standings(self.league_standings, team1_clean)
                    
                    match_obj = Match(
                        date=date.strip() if date else "",
                        home_team=team1_clean,
                        away_team=team2_clean,
                        score=f"{g1_int}-{g2_int}",
                        result=result,
                        goals_scored=goals_scored,
                        goals_conceded=goals_conceded,
                        venue=venue,
                        competition=competition.strip() if 'competition' in locals() else "Unknown",
                        opponent_position=opponent_position
                    )
                    matches.append(match_obj)
                    
            except Exception as e:
                continue
        
        # Sort by date (newest first)
        matches.sort(key=lambda x: self.parse_date_to_sortable(x.date), reverse=True)
        
        # Remove duplicates
        unique_matches = []
        seen = set()
        for match in matches:
            key = (match.date, match.home_team, match.away_team, match.score)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        # Return only last 6 matches
        return unique_matches[:6]
    
    def parse_date_to_sortable(self, date_str: str):
        """Parse date to sortable format"""
        if not date_str:
            return '00000000'
        
        date_patterns = [
            r'(\d{2})/(\d{2})/(\d{4})',
            r'(\d{2})/(\d{2})/(\d{2})',
            r'(\d{2})/(\d{2})',
        ]
        
        for pattern in date_patterns:
            match = re.match(pattern, date_str)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    day, month, year = groups
                    if len(year) == 2:
                        year = '20' + year
                    return f"{year}{month.zfill(2)}{day.zfill(2)}"
                elif len(groups) == 2:
                    day, month = groups
                    current_year = datetime.now().year
                    return f"{current_year}{month.zfill(2)}{day.zfill(2)}"
        
        return '00000000'
    
    def extract_competition_name(self, soup: BeautifulSoup):
        """Extract competition name from Forebet page"""
        if not soup:
            return "Unknown Competition"
        
        # Try breadcrumbs
        breadcrumb_selectors = [
            'div.breadcrumb a',
            'nav.breadcrumb a',
            '.breadcrumbs a',
            '.breadcrumb a',
            'div.breadcrumbs a',
            'ul.breadcrumb li a',
            'span.breadcrumb a',
            'a.breadcrumb'
        ]
        
        for selector in breadcrumb_selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text(strip=True)
                if text and len(text) > 3:
                    text_lower = text.lower()
                    if text_lower not in ['home', 'football', 'predictions', 'matches', 'forebet']:
                        if any(keyword in text_lower for keyword in ['league', 'cup', 'championship', 'division', 'liga', 'premier', 'serie', 'bundes']):
                            return text
        
        # Try page title
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text()
            patterns = [
                r'Prediction\s*[-\–]\s*([^\-]+)',
                r'-\s*([^-]+)\s+Prediction',
                r'([A-Za-z\s]+)\s+-\s+\d'
            ]
            for pattern in patterns:
                match = re.search(pattern, title_text, re.IGNORECASE)
                if match:
                    comp = match.group(1).strip()
                    if comp and len(comp) > 3:
                        return comp
        
        # Try to find competition in headings
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        for heading in headings:
            text = heading.get_text(strip=True)
            if text and any(keyword in text.lower() for keyword in ['league', 'cup', 'championship', 'division']):
                return text
        
        return "Unknown Competition"
    
    def calculate_probabilities(self, home_metrics: TeamMetrics, away_metrics: TeamMetrics):
        """Calculate match probabilities based on computed metrics"""
        
        # Start with base scores
        home_score = 0
        away_score = 0
        
        # 1. Form rating (20%)
        home_score += home_metrics.form_rating * self.factors.form_weight
        away_score += away_metrics.form_rating * self.factors.form_weight
        
        # 2. Attack vs Defense (25%)
        # Home attack vs away defense
        home_attack_vs_defense = (home_metrics.goal_scoring_rate * 0.7 + 
                                 (1 - away_metrics.goal_conceding_rate) * 0.3)
        away_attack_vs_defense = (away_metrics.goal_scoring_rate * 0.7 + 
                                 (1 - home_metrics.goal_conceding_rate) * 0.3)
        
        home_score += home_attack_vs_defense * self.factors.attack_vs_defense_weight
        away_score += away_attack_vs_defense * self.factors.attack_vs_defense_weight
        
        # 3. Opponent quality (15%)
        # Teams that performed well against better opponents get advantage
        home_opponent_quality = home_metrics.performance_vs_top6 * 0.6 + (1 - home_metrics.avg_opponent_position/20) * 0.4
        away_opponent_quality = away_metrics.performance_vs_top6 * 0.6 + (1 - away_metrics.avg_opponent_position/20) * 0.4
        
        home_score += home_opponent_quality * self.factors.opponent_quality_weight
        away_score += away_opponent_quality * self.factors.opponent_quality_weight
        
        # 4. Home advantage (10%) - from recent form
        home_score += max(0, home_metrics.home_advantage) * self.factors.home_advantage_weight
        away_score += max(0, -home_metrics.home_advantage) * self.factors.home_advantage_weight
        
        # 5. Momentum (10%)
        home_momentum = (home_metrics.momentum + 1) / 2  # Convert -1 to 1 range to 0 to 1
        away_momentum = (away_metrics.momentum + 1) / 2
        
        home_score += home_momentum * self.factors.momentum_weight
        away_score += away_momentum * self.factors.momentum_weight
        
        # 6. Consistency (10%)
        # Based on clean sheet rate and goal trends
        home_consistency = (home_metrics.clean_sheet_rate * 0.6 + 
                           (1 - abs(home_metrics.goal_trend)) * 0.2 +
                           (1 - abs(home_metrics.defense_trend)) * 0.2)
        away_consistency = (away_metrics.clean_sheet_rate * 0.6 + 
                           (1 - abs(away_metrics.goal_trend)) * 0.2 +
                           (1 - abs(away_metrics.defense_trend)) * 0.2)
        
        home_score += home_consistency * self.factors.consistency_weight
        away_score += away_consistency * self.factors.consistency_weight
        
        # 7. Top6 Performance (5%)
        home_score += (home_metrics.top6_performance_score / 10) * self.factors.top6_performance_weight
        away_score += (away_metrics.top6_performance_score / 10) * self.factors.top6_performance_weight
        
        # 8. H2H factor (10%) - if available
        if home_metrics.h2h_metrics.total_matches >= 3:
            h2h_home_advantage = 0
            
            if home_metrics.h2h_metrics.home_win_rate > home_metrics.h2h_metrics.away_win_rate + 0.2:
                h2h_home_advantage = 0.3  # Strong home H2H advantage
            elif home_metrics.h2h_metrics.away_win_rate > home_metrics.h2h_metrics.home_win_rate + 0.2:
                h2h_home_advantage = -0.3  # Strong away H2H advantage
            elif home_metrics.h2h_metrics.draw_rate > 0.4:
                # Draw tendency - slightly reduces both teams' scores
                h2h_home_advantage = 0
                home_score *= 0.9
                away_score *= 0.9
            
            home_score += (0.5 + h2h_home_advantage) * self.factors.h2h_weight
            away_score += (0.5 - h2h_home_advantage) * self.factors.h2h_weight
        
        # Base draw probability (draws more likely when teams are evenly matched)
        closeness = 1 - abs(home_score - away_score)
        base_draw = 0.25 * closeness
        
        # Normalize to probabilities
        total = home_score + away_score + base_draw
        home_prob = home_score / total * 100
        away_prob = away_score / total * 100
        draw_prob = base_draw / total * 100
        
        return {
            'home_win': home_prob,
            'away_win': away_prob,
            'draw': draw_prob,
            'home_score': home_score,
            'away_score': away_score
        }
    
    def calculate_goal_probabilities(self, home_metrics: TeamMetrics, away_metrics: TeamMetrics):
        """Calculate goal market probabilities based on computed metrics"""
        
        # Expected goals based on scoring rates and defense
        home_expected = (home_metrics.goal_scoring_rate * 0.7 + 
                        (1 - away_metrics.clean_sheet_rate) * 0.3)
        away_expected = (away_metrics.goal_scoring_rate * 0.7 + 
                        (1 - home_metrics.clean_sheet_rate) * 0.3)
        
        # Adjust for form and momentum
        home_expected *= (0.8 + home_metrics.form_rating * 0.4)
        away_expected *= (0.8 + away_metrics.form_rating * 0.4)
        
        # Apply momentum adjustments
        home_expected *= (1 + home_metrics.goal_trend * 0.3)
        away_expected *= (1 + away_metrics.goal_trend * 0.3)
        
        # Adjust for H2H goal patterns if available
        if home_metrics.h2h_metrics.total_matches >= 3:
            h2h_weight = 0.15
            h2h_home_goals = home_metrics.h2h_metrics.home_goals_for
            h2h_away_goals = home_metrics.h2h_metrics.away_goals_for
            
            home_expected = home_expected * (1 - h2h_weight) + h2h_home_goals * h2h_weight
            away_expected = away_expected * (1 - h2h_weight) + h2h_away_goals * h2h_weight
        
        # Ensure reasonable values
        home_expected = max(0.2, min(4.0, home_expected))
        away_expected = max(0.2, min(3.0, away_expected))
        
        # Poisson distribution calculations
        def poisson_prob(lambda_, k):
            if k > 8:
                return 0
            return (lambda_ ** k * math.exp(-lambda_)) / math.factorial(k)
        
        over_25_prob = 0
        bts_prob = 0
        
        max_goals = 7
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = poisson_prob(home_expected, i) * poisson_prob(away_expected, j)
                if i + j > 2.5:
                    over_25_prob += prob
                if i > 0 and j > 0:
                    bts_prob += prob
        
        home_clean_sheet = math.exp(-away_expected)
        away_clean_sheet = math.exp(-home_expected)
        
        return {
            'over_25': over_25_prob * 100,
            'under_25': (1 - over_25_prob) * 100,
            'bts': bts_prob * 100,
            'expected_home_goals': home_expected,
            'expected_away_goals': away_expected,
            'total_expected_goals': home_expected + away_expected,
            'home_clean_sheet': home_clean_sheet * 100,
            'away_clean_sheet': away_clean_sheet * 100
        }
    
    def analyze_match(self, url: str):
        """Main analysis function that returns JSON results"""
        
        result = {
            'success': False,
            'error': None,
            'data': None
        }
        
        try:
            # Fetch the page
            soup = self.fetch_page_soup(url)
            
            if not soup:
                result['error'] = "Failed to fetch page"
                return result
            
            # Extract basic information
            home_team, away_team, home_position, away_position = self.extract_team_names_and_positions(soup)
            
            if not home_team or not away_team:
                result['error'] = "Could not extract team names"
                return result
            
            # Extract competition
            competition = self.extract_competition_name(soup)
            
            # Extract H2H data
            h2h_matches = self.extract_head_to_head_matches(soup, home_team, away_team)
            
            # Calculate H2H metrics
            h2h_metrics = self.calculate_h2h_metrics(h2h_matches, home_team, away_team)
            
            # Extract recent matches
            html = str(soup)
            home_matches = self.get_last_six_matches(html, home_team, competition)
            away_matches = self.get_last_six_matches(html, away_team, competition)
            
            if len(home_matches) == 0 or len(away_matches) == 0:
                result['error'] = "No recent match data found"
                return result
            
            # Compute metrics
            home_metrics = self.decision_engine.compute_metrics_from_matches(
                home_team, home_matches, True, home_position, h2h_metrics
            )
            away_metrics = self.decision_engine.compute_metrics_from_matches(
                away_team, away_matches, False, away_position, H2HMetrics()
            )
            
            # Analyze core factors
            core_analysis = self.decision_engine.analyze_core_factors(home_metrics, away_metrics)
            
            # Calculate probabilities
            probabilities = self.calculate_probabilities(home_metrics, away_metrics)
            
            # Calculate goal markets
            goal_markets = self.calculate_goal_probabilities(home_metrics, away_metrics)
            
            # Analyze match summary
            match_summary = self.decision_engine.analyze_match_summary(
                probabilities, goal_markets, home_metrics, away_metrics, core_analysis
            )
            
            # Make decisions
            decisions = self.decision_engine.make_decisions(
                probabilities, goal_markets, home_metrics, away_metrics, 
                core_analysis, match_summary
            )
            
            # Prepare response data
            result['success'] = True
            result['data'] = {
                'match': {
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_position': home_position,
                    'away_position': away_position,
                    'competition': competition
                },
                'h2h': {
                    'total_matches': h2h_metrics.total_matches,
                    'home_wins': h2h_metrics.home_wins,
                    'away_wins': h2h_metrics.away_wins,
                    'draws': h2h_metrics.draws,
                    'home_win_rate': h2h_metrics.home_win_rate,
                    'away_win_rate': h2h_metrics.away_win_rate,
                    'draw_rate': h2h_metrics.draw_rate,
                    'recent_matches': [
                        {
                            'date': m.date,
                            'home_team': m.home_team,
                            'away_team': m.away_team,
                            'score': f"{m.home_goals}-{m.away_goals}",
                            'competition': m.competition
                        }
                        for m in h2h_matches[:5]
                    ]
                },
                'metrics': {
                    'home': {
                        'form_rating': home_metrics.form_rating,
                        'goal_scoring_rate': home_metrics.goal_scoring_rate,
                        'goal_conceding_rate': home_metrics.goal_conceding_rate,
                        'clean_sheet_rate': home_metrics.clean_sheet_rate,
                        'top6_opponents_faced': home_metrics.top6_opponents_faced,
                        'performance_vs_top6': home_metrics.performance_vs_top6,
                        'top6_performance_score': home_metrics.top6_performance_score,
                        'avg_opponent_position': home_metrics.avg_opponent_position,
                        'momentum': home_metrics.momentum
                    },
                    'away': {
                        'form_rating': away_metrics.form_rating,
                        'goal_scoring_rate': away_metrics.goal_scoring_rate,
                        'goal_conceding_rate': away_metrics.goal_conceding_rate,
                        'clean_sheet_rate': away_metrics.clean_sheet_rate,
                        'top6_opponents_faced': away_metrics.top6_opponents_faced,
                        'performance_vs_top6': away_metrics.performance_vs_top6,
                        'top6_performance_score': away_metrics.top6_performance_score,
                        'avg_opponent_position': away_metrics.avg_opponent_position,
                        'momentum': away_metrics.momentum
                    }
                },
                'probabilities': probabilities,
                'goal_markets': goal_markets,
                'analysis': {
                    'form_difference': core_analysis.form_difference,
                    'goal_scoring_difference': core_analysis.goal_scoring_difference,
                    'goal_conceding_difference': core_analysis.goal_conceding_difference,
                    'top6_performance_difference': core_analysis.top6_performance_difference,
                    'h2h_dominance': core_analysis.h2h_dominance,
                    'key_factors': core_analysis.key_factors
                },
                'decisions': decisions
            }
            
        except Exception as e:
            result['error'] = str(e)
        
        return result

# Flask App for Render
app = Flask(__name__)
analyzer = ForebetAnalyzer()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Football Match Analyzer</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4a6fa5 0%, #2c3e50 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .content {
            padding: 30px;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #4a6fa5;
        }
        
        button {
            background: linear-gradient(135deg, #4a6fa5 0%, #2c3e50 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            display: block;
            width: 100%;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #4a6fa5;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .match-header {
            background: #f0f7ff;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .match-teams {
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .competition {
            color: #666;
            font-size: 1.1em;
        }
        
        .section {
            margin-bottom: 30px;
        }
        
        .section-title {
            font-size: 1.4em;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .bet-recommendation {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #28a745;
        }
        
        .bet-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #155724;
            margin-bottom: 10px;
        }
        
        .confidence-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .confidence-high {
            background: #28a745;
            color: white;
        }
        
        .confidence-medium {
            background: #ffc107;
            color: #333;
        }
        
        .confidence-low {
            background: #dc3545;
            color: white;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #dc3545;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
            }
            
            .header {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 1.8em;
            }
            
            .content {
                padding: 20px;
            }
            
            .metric-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚽ Football Match Analyzer</h1>
            <p>Data-driven predictions and betting insights from Forebet.com</p>
        </div>
        
        <div class="content">
            <form method="POST" action="/">
                <div class="form-group">
                    <label for="url">Forebet Match URL:</label>
                    <input type="text" id="url" name="url" 
                           placeholder="https://www.forebet.com/en/predictions-tips/manchester-city-liverpool"
                           value="{{ url if url else '' }}" required>
                </div>
                <button type="submit">🔍 Analyze Match</button>
            </form>
            
            {% if result %}
                {% if result.success %}
                    <div class="result">
                        <div class="match-header">
                            <div class="match-teams">
                                {{ result.data.match.home_team }} vs {{ result.data.match.away_team }}
                            </div>
                            <div class="competition">
                                {{ result.data.match.competition }}
                            </div>
                        </div>
                        
                        <div class="section">
                            <div class="section-title">📊 Match Probabilities</div>
                            <div class="metric-grid">
                                <div class="metric-card">
                                    <div class="metric-value">{{ "%.1f"|format(result.data.probabilities.home_win) }}%</div>
                                    <div class="metric-label">{{ result.data.match.home_team }} Win</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{{ "%.1f"|format(result.data.probabilities.draw) }}%</div>
                                    <div class="metric-label">Draw</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{{ "%.1f"|format(result.data.probabilities.away_win) }}%</div>
                                    <div class="metric-label">{{ result.data.match.away_team }} Win</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="section">
                            <div class="section-title">🎯 Key Factors</div>
                            <ul>
                                {% for factor in result.data.analysis.key_factors %}
                                    <li style="margin-bottom: 8px;">• {{ factor }}</li>
                                {% endfor %}
                            </ul>
                        </div>
                        
                        {% if result.data.decisions.primary_bet %}
                        <div class="bet-recommendation">
                            <div class="bet-title">💡 Primary Bet Recommendation</div>
                            <p><strong>Selection:</strong> 
                                {% if result.data.decisions.primary_bet.selection == 'HOME_WIN' %}
                                    {{ result.data.match.home_team }} Win
                                {% elif result.data.decisions.primary_bet.selection == 'AWAY_WIN' %}
                                    {{ result.data.match.away_team }} Win
                                {% else %}
                                    Draw
                                {% endif %}
                            </p>
                            <p><strong>Probability:</strong> {{ "%.1f"|format(result.data.decisions.primary_bet.probability) }}%</p>
                            <p><strong>Odds Range:</strong> {{ result.data.decisions.primary_bet.odds_range }}</p>
                            <div class="confidence-badge confidence-{{ result.data.decisions.confidence|lower }}">
                                Confidence: {{ result.data.decisions.confidence }} ({{ result.data.decisions.confidence_score }}/10)
                            </div>
                            <p><strong>Stake Suggestion:</strong> {{ result.data.decisions.primary_bet.stake_suggestion }}</p>
                            {% if result.data.decisions.primary_bet.detailed_reasoning %}
                                <p><strong>Reasoning:</strong> {{ result.data.decisions.primary_bet.detailed_reasoning }}</p>
                            {% endif %}
                        </div>
                        {% endif %}
                        
                        <div class="section">
                            <div class="section-title">📈 Goal Market Analysis</div>
                            <div class="metric-grid">
                                <div class="metric-card">
                                    <div class="metric-value">{{ "%.1f"|format(result.data.goal_markets.over_25) }}%</div>
                                    <div class="metric-label">Over 2.5 Goals</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{{ "%.1f"|format(result.data.goal_markets.bts) }}%</div>
                                    <div class="metric-label">Both Teams Score</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-value">{{ "%.2f"|format(result.data.goal_markets.total_expected_goals) }}</div>
                                    <div class="metric-label">Expected Total Goals</div>
                                </div>
                            </div>
                        </div>
                        
                        {% if result.data.h2h.total_matches > 0 %}
                        <div class="section">
                            <div class="section-title">🤝 Head-to-Head History</div>
                            <p><strong>Record:</strong> {{ result.data.h2h.home_wins }}W - {{ result.data.h2h.draws }}D - {{ result.data.h2h.away_wins }}L</p>
                            <p><strong>Win Rates:</strong> Home {{ "%.0f"|format(result.data.h2h.home_win_rate * 100) }}% | 
                               Draw {{ "%.0f"|format(result.data.h2h.draw_rate * 100) }}% | 
                               Away {{ "%.0f"|format(result.data.h2h.away_win_rate * 100) }}%</p>
                        </div>
                        {% endif %}
                    </div>
                {% else %}
                    <div class="error">
                        <strong>❌ Error:</strong> {{ result.error }}
                    </div>
                {% endif %}
            {% elif loading %}
                <div class="loading">
                    <p>⏳ Analyzing match data... This may take up to 30 seconds.</p>
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        if url:
            result = analyzer.analyze_match(url)
            return render_template_string(HTML_TEMPLATE, url=url, result=result, loading=False)
    
    return render_template_string(HTML_TEMPLATE, url='', result=None, loading=False)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required'}), 400
    
    url = data['url'].strip()
    result = analyzer.analyze_match(url)
    
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'football-analyzer'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
