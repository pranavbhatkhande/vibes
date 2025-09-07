#!/usr/bin/env python3
"""
Mastermind Solver using MLX
An AI-powered Mastermind codebreaker that uses MLX for high-performance computation
to find optimal guesses and beat human players consistently.
"""

import mlx.core as mx
import itertools
import random
from typing import List, Tuple, Set
import time

# Enable MLX GPU/ANE acceleration
mx.set_default_device(mx.gpu)  # Use GPU/Apple Neural Engine if available

class MastermindSolver:
    def __init__(self, num_colors: int = 6, code_length: int = 4):
        self.num_colors = num_colors
        self.code_length = code_length
        self.colors = list(range(num_colors))  # 0 to num_colors-1

        # Color mapping for user-friendly display
        self.color_names = {
            0: "⚫ Black",
            1: "⚪ White", 
            2: "🔴 Red",
            3: "🟢 Green",
            4: "🟡 Yellow",
            5: "🔵 Blue"
        }

        # Generate all possible codes using MLX for efficiency
        print("🔧 Initializing Mastermind Solver...")
        print(f"🎨 Colors: {num_colors}, Code Length: {code_length}")
        print("Color mapping:", ", ".join([f"{i}: {self.color_names[i]}" for i in range(min(num_colors, len(self.color_names)))]))

        # Create all possible combinations
        all_codes = list(itertools.product(self.colors, repeat=code_length))
        self.all_codes = mx.array(all_codes, dtype=mx.int32)
        self.possible_codes = set(range(len(all_codes)))

        print(f"📊 Total possible codes: {len(self.all_codes)}")
        print(f"🚀 Using device: {mx.default_device()}")

        # Pre-compute all possible feedback patterns for optimization
        self._precompute_feedback_patterns()

        print("✅ Solver initialized with hardware acceleration!")

        # For testing/simulation
        self.secret_code = None
        self.game_stats = {
            'games_played': 0,
            'total_guesses': 0,
            'wins': 0,
            'losses': 0,
            'guess_distribution': {}
        }

    def _precompute_feedback_patterns(self):
        """Pre-compute all possible feedback patterns for optimization."""
        self.feedback_patterns = []
        for red in range(self.code_length + 1):
            for white in range(self.code_length - red + 1):
                if red + white <= self.code_length:
                    self.feedback_patterns.append((red, white))
        print(f"📋 Pre-computed {len(self.feedback_patterns)} feedback patterns")

    def calculate_score(self, guess: List[int], secret: List[int]) -> Tuple[int, int]:
        """Calculate red and white pins for a guess against the secret code using efficient counting."""
        red_pins = 0
        white_pins = 0

        # Count frequencies
        guess_counts = [0] * self.num_colors
        secret_counts = [0] * self.num_colors
        used_secret = [False] * self.code_length

        # First pass: count red pins and mark used positions
        for i in range(self.code_length):
            if guess[i] == secret[i]:
                red_pins += 1
                used_secret[i] = True
            else:
                guess_counts[guess[i]] += 1
                secret_counts[secret[i]] += 1

        # Second pass: count white pins
        for color in range(self.num_colors):
            white_pins += min(guess_counts[color], secret_counts[color])

        return red_pins, white_pins

    def get_best_guess(self) -> List[int]:
        """Find the best guess using Donald Knuth's minimax algorithm with MLX acceleration."""
        if len(self.possible_codes) == 1:
            return self.all_codes[list(self.possible_codes)[0]].tolist()

        if len(self.possible_codes) <= 2:
            return self.all_codes[list(self.possible_codes)[0]].tolist()

        # Special case: First guess should be [0,0,1,1] (Knuth's optimal first guess)
        if len(self.possible_codes) == len(self.all_codes):
            first_guess = [0, 0, 1, 1]
            return first_guess

        # Use Knuth's minimax algorithm
        return self._knuth_minimax_guess()

    def _knuth_minimax_guess(self) -> List[int]:
        """Implement Knuth's minimax algorithm with MLX optimization."""
        best_guess = None
        best_score = float('inf')

        # Smart candidate selection for performance
        candidates = self._select_smart_candidates()

        print(f"🔍 Evaluating {len(candidates)} candidate guesses with MLX acceleration...")

        start_time = time.time()
        for candidate_idx in candidates:
            candidate = self.all_codes[candidate_idx].tolist()

            # Calculate the minimax score for this candidate
            max_remaining = self._calculate_max_remaining_optimized(candidate)

            if max_remaining < best_score:
                best_score = max_remaining
                best_guess = candidate

        eval_time = time.time() - start_time
        print(".2f")

        if best_guess is None:
            # Fallback to first possibility
            best_guess = self.all_codes[list(self.possible_codes)[0]].tolist()

        print(f"🎯 Best minimax score: {best_score} remaining possibilities")
        return best_guess

    def _select_smart_candidates(self) -> List[int]:
        """Select the most promising candidates for evaluation."""
        if len(self.possible_codes) <= 20:
            return list(self.possible_codes)

        # For larger sets, use a combination of strategies
        candidates = []

        # Always include remaining possibilities (prioritize these)
        candidates.extend(list(self.possible_codes))

        # Add some diverse codes to ensure good coverage
        diverse_indices = self._select_diverse_codes(min(30, len(self.all_codes)))
        candidates.extend(diverse_indices)

        # Remove duplicates and limit total
        candidates = list(set(candidates))[:min(50, len(candidates))]

        return candidates

    def _select_diverse_codes(self, num_codes: int) -> List[int]:
        """Select diverse codes for better minimax evaluation."""
        if num_codes >= len(self.all_codes):
            return list(range(len(self.all_codes)))

        # Simple diversity: select evenly spaced indices
        step = max(1, len(self.all_codes) // num_codes)
        return [i * step for i in range(num_codes)]

    def _calculate_max_remaining_optimized(self, guess: List[int]) -> int:
        """Calculate the maximum number of possibilities that could remain after any feedback (optimized)."""
        max_remaining = 0

        # Use pre-computed feedback patterns for speed
        for red_pins, white_pins in self.feedback_patterns:
            if red_pins + white_pins > self.code_length:
                continue

            # Count how many current possibilities would give this feedback
            remaining_count = self._count_codes_with_feedback(guess, red_pins, white_pins)

            if remaining_count > max_remaining:
                max_remaining = remaining_count

        return max_remaining

    def _count_codes_with_feedback(self, guess: List[int], target_red: int, target_white: int) -> int:
        """Count how many possible codes would give the specified feedback for this guess using MLX vectorization."""
        if not self.possible_codes:
            return 0

        # Convert guess to MLX array
        guess_arr = mx.array(guess)

        # Get all possible codes as MLX array
        possible_indices = list(self.possible_codes)
        possible_codes = self.all_codes[possible_indices]

        # Calculate red and white pins for ALL codes simultaneously using vectorized operations
        red_pins_all, white_pins_all = self._calculate_scores_vectorized(guess_arr, possible_codes)

        # Count codes that match the target feedback
        matches = (red_pins_all == target_red) & (white_pins_all == target_white)
        count = mx.sum(matches).item()

        return count

    def _calculate_scores_vectorized(self, guess_arr: mx.array, codes_batch: mx.array) -> Tuple[mx.array, mx.array]:
        """Calculate red and white pins for all codes in batch using JIT-compiled MLX operations."""
        return _calculate_scores_vectorized_compiled(guess_arr, codes_batch, self.num_colors)

    def _evaluate_guess(self, guess: List[int]) -> float:
        """Evaluate how good a guess is by simulating all possible outcomes."""
        score_distribution = {}

        for secret_idx in list(self.possible_codes)[:50]:  # Sample for speed
            secret = self.all_codes[secret_idx].tolist()
            score = self.calculate_score(guess, secret)
            score_distribution[score] = score_distribution.get(score, 0) + 1

        # Higher score for guesses that create more balanced distributions
        if not score_distribution:
            return 0

        max_count = max(score_distribution.values())
        total = sum(score_distribution.values())

        # Prefer guesses that don't leave too many possibilities
    def update_possible_codes(self, guess: List[int], red_pins: int, white_pins: int):
        """Update the set of possible codes based on feedback."""
        guess_mx = mx.array(guess, dtype=mx.int32)
        new_possible = set()

        for idx in self.possible_codes:
            candidate = self.all_codes[idx]
            score = self.calculate_score(guess, candidate.tolist())
            if score == (red_pins, white_pins):
                new_possible.add(idx)

        self.possible_codes = new_possible
        print(f"🔍 Possible codes remaining: {len(self.possible_codes)}")

    def play_game(self):
        """Interactive game loop with user-friendly color display."""
        print("🎯 Welcome to Mastermind AI Solver!")
        print("🎨 Using 6 colors: Black, White, Red, Green, Yellow, Blue")
        print("Color mapping:")
        for i in range(self.num_colors):
            if i in self.color_names:
                print(f"  {i}: {self.color_names[i]}")
        print("\nExample: If your code is Black, White, Red, Green, enter: 0,1,2,3")
        print("Example: If your code is Yellow, Blue, Yellow, Blue, enter: 4,5,4,5")
        print("=" * 60)

        guesses = 0
        max_guesses = 10

        while guesses < max_guesses:
            guesses += 1

            if len(self.possible_codes) == 0:
                print("❌ No possible codes left! Something went wrong.")
                return

            # Get best guess
            start_time = time.time()
            guess = self.get_best_guess()
            end_time = time.time()

            print(f"\n🤖 Guess #{guesses}:")
            print(f"   Array: {guess}")
            print(f"   Colors: {self.format_code(guess)}")
            print(f"   Compact: {self.format_code_compact(guess)}")
            print(".3f")
            print(f"   Possible codes remaining: {len(self.possible_codes)}")

            if len(self.possible_codes) == 1:
                print("🎉 I found your code!")
                return

            # Get user feedback
            try:
                print("\n📊 Feedback for this guess:")
                red = int(input("   🔴 Red pins (correct color AND position): "))
                white = int(input("   ⚪ White pins (correct color, wrong position): "))

                if red == self.code_length:
                    print("🎉 Correct! I cracked your code!")
                    return

                # Update possibilities
                self.update_possible_codes(guess, red, white)

            except ValueError:
                print("❌ Please enter valid numbers for red and white pins.")
                guesses -= 1  # Don't count invalid input
                continue

        print("😅 I couldn't crack your code within the limit. You win this time!")

    def simulate_game(self, secret_code: List[int] = None, max_guesses: int = 10, verbose: bool = True) -> Tuple[bool, int]:
        """Simulate a complete game with automatic feedback."""
        # Reset solver state
        self.possible_codes = set(range(len(self.all_codes)))

        # Set secret code
        if secret_code is None:
            self.secret_code = random.choice(self.all_codes).tolist()
        else:
            self.secret_code = secret_code

        if verbose:
            print(f"🎯 Simulating game with secret code: {self.format_code(self.secret_code)}")

        guesses = 0
        game_won = False

        while guesses < max_guesses and not game_won:
            guesses += 1

            if len(self.possible_codes) == 0:
                if verbose:
                    print("❌ No possible codes left! Logic error.")
                break

            # Get AI guess
            guess = self.get_best_guess()

            if verbose:
                print(f"\n🤖 Guess #{guesses}:")
                print(f"   Array: {guess}")
                print(f"   Colors: {self.format_code(guess)}")
                print(f"   Compact: {self.format_code_compact(guess)}")

            # Get automatic feedback
            red_pins, white_pins = self.calculate_score(guess, self.secret_code)

            if verbose:
                print(f"📊 Feedback: {red_pins} red, {white_pins} white")

            if red_pins == self.code_length:
                game_won = True
                if verbose:
                    print("🎉 Code cracked!")
                break

            # Update possibilities
            self.update_possible_codes(guess, red_pins, white_pins)

        if verbose:
            if game_won:
                print(f"✅ Won in {guesses} guesses!")
            else:
                print(f"❌ Failed after {max_guesses} guesses. Secret was {self.secret_code}")

        return game_won, guesses

    def run_automated_tests(self, num_games: int = 100, max_guesses: int = 10, verbose: bool = False):
        """Run automated tests to evaluate solver performance."""
        print(f"🧪 Running {num_games} automated tests...")
        print("=" * 50)

        start_time = time.time()

        for game in range(num_games):
            if verbose or (game + 1) % 10 == 0:
                print(f"Game {game + 1}/{num_games}")

            won, guesses = self.simulate_game(max_guesses=max_guesses, verbose=verbose)

            # Update statistics
            self.game_stats['games_played'] += 1
            self.game_stats['total_guesses'] += guesses

            if won:
                self.game_stats['wins'] += 1
            else:
                self.game_stats['losses'] += 1

            # Track guess distribution
            self.game_stats['guess_distribution'][guesses] = \
                self.game_stats['guess_distribution'].get(guesses, 0) + 1

    def format_code(self, code: List[int]) -> str:
        """Format a code array into a user-friendly string with colors."""
        if isinstance(code, mx.array):
            code = code.tolist()

        color_parts = []
        for color_num in code:
            if color_num in self.color_names:
                # Extract just the emoji and color name
                emoji_and_name = self.color_names[color_num]
                color_parts.append(emoji_and_name)
            else:
                color_parts.append(f"Color {color_num}")

        return " | ".join(color_parts)

    def format_code_compact(self, code: List[int]) -> str:
        """Format a code array into a compact string with emojis only."""
        if isinstance(code, mx.array):
            code = code.tolist()

        emojis = []
        for color_num in code:
            if color_num in self.color_names:
                # Extract just the emoji
                emoji = self.color_names[color_num].split()[0]
                emojis.append(emoji)
            else:
                emojis.append(str(color_num))

        return "".join(emojis)

        end_time = time.time()

        # Print results
        self._print_test_results(end_time - start_time)

    def _print_test_results(self, total_time: float):
        """Print comprehensive test results."""
        print("\n📊 TEST RESULTS")
        print("=" * 50)

        games = self.game_stats['games_played']
        wins = self.game_stats['wins']
        losses = self.game_stats['losses']
        total_guesses = self.game_stats['total_guesses']

        print(f"🎮 Games Played: {games}")
        print(f"🏆 Win Rate: {wins/games*100:.1f}% ({wins}/{games})")
        print(f"📈 Average Guesses: {total_guesses/games:.2f}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"⚡ Games per Second: {games/total_time:.2f}")

        print("\n📋 Guess Distribution:")
        for guesses in sorted(self.game_stats['guess_distribution'].keys()):
            count = self.game_stats['guess_distribution'][guesses]
            percentage = count / games * 100
            print(f"  {guesses} guesses: {count} games ({percentage:.1f}%)")

        # Edge case analysis
        print("\n🔍 Edge Case Analysis:")
        if 1 in self.game_stats['guess_distribution']:
            lucky_guesses = self.game_stats['guess_distribution'][1]
            print(f"🍀 Lucky first guesses: {lucky_guesses} ({lucky_guesses/games*100:.1f}%)")

        max_guesses = max(self.game_stats['guess_distribution'].keys())
        hard_games = sum(count for g, count in self.game_stats['guess_distribution'].items() if g >= 8)
        print(f"🎯 Hard games (≥8 guesses): {hard_games} ({hard_games/games*100:.1f}%)")

    def test_edge_cases(self):
        """Test specific edge cases that might break the solver."""
        print("🧪 Testing Edge Cases...")
        print("=" * 30)

        edge_cases = [
            ([0, 0, 0, 0], "All Black (most common guess)"),
            ([0, 1, 2, 3], "Black, White, Red, Green (sequential)"),
            ([5, 5, 5, 5], "All Blue"),
            ([0, 0, 1, 1], "Black-Black-White-White (pairs)"),
            ([0, 1, 0, 1], "Black-White-Black-White (alternating)"),
            ([1, 2, 3, 4], "White, Red, Green, Yellow (no Black)"),
            ([2, 3, 4, 5], "Red, Green, Yellow, Blue (no Black/White)"),
            ([4, 2, 5, 0], "Yellow, Red, Blue, Black (mixed)"),
        ]

        for secret, description in edge_cases:
            print(f"\n🎯 Testing: {description}")
            print(f"   Secret: {self.format_code(secret)}")

            # Reset solver
            self.possible_codes = set(range(len(self.all_codes)))

            won, guesses = self.simulate_game(secret_code=secret, max_guesses=15, verbose=False)

            if won:
                print(f"   ✅ Solved in {guesses} guesses")
            else:
                print(f"   ❌ Failed after {guesses} guesses")

        print("\n✅ Edge case testing complete!")

# JIT compile critical functions for maximum performance
def _calculate_scores_vectorized_jit(guess_arr: mx.array, codes_batch: mx.array, num_colors: int) -> Tuple[mx.array, mx.array]:
    """JIT-compiled version of score calculation using pure MLX operations."""
    num_codes = codes_batch.shape[0]

    # Calculate red pins (exact matches) for all codes at once
    red_pins = mx.sum(guess_arr == codes_batch, axis=1)

    # Calculate white pins using vectorized operations
    white_pins = mx.zeros(num_codes, dtype=mx.int32)

    # For each color, count minimum matches in non-red positions
    for color in range(num_colors):
        guess_count = mx.sum(guess_arr == color)
        code_counts = mx.sum(codes_batch == color, axis=1)
        white_pins += mx.minimum(guess_count, code_counts)

    # Subtract red pins from white pins (since red pins are already counted in white)
    white_pins -= red_pins

    return red_pins, white_pins

# Compile the function
_calculate_scores_vectorized_compiled = mx.compile(_calculate_scores_vectorized_jit)

def main():
    print("🚀 MLX-Powered Mastermind Solver")
    print("=" * 50)

    # Game configuration - Standard Mastermind: 6 colors, 4 positions
    solver = MastermindSolver(num_colors=6, code_length=4)

    print("Choose mode:")
    print("1. 🎮 Play interactive game (you provide feedback)")
    print("2. 🧪 Run automated tests")
    print("3. 🔍 Test edge cases")
    print("4. 🎯 Simulate single game")

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        solver.play_game()
    elif choice == "2":
        num_games = int(input("Number of games to simulate (default 100): ") or "100")
        solver.run_automated_tests(num_games=num_games, verbose=False)
    elif choice == "3":
        solver.test_edge_cases()
    elif choice == "4":
        print("I'll simulate a game and show you the process...")
        solver.simulate_game(verbose=True)
    else:
        print("Invalid choice. Starting interactive game...")
        solver.play_game()

    print("\n👋 Thanks for using Mastermind Solver!")

if __name__ == "__main__":
    main()
