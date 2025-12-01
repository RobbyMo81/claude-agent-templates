
import { PrizeTier } from './types';

export const REQUIRED_COLUMNS: string[] = ['draw_date', 'wb1', 'wb2', 'wb3', 'wb4', 'wb5', 'pb'];

export const PRIZE_TIER_ORDER: PrizeTier[] = [
    "Grand Prize",
    "$1 Million",
    "$50,000",
    "$100 (4+0)",
    "$100 (3+1)",
    "$7 (3+0)",
    "$7 (2+1)",
    "$4 (1+1)",
    "$4 (0+1)",
    "No Win",
];
