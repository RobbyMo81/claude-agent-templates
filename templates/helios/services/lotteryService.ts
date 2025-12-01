
import { BacktestResults, HistoricalDraw, PrizeTier, Ticket } from '../types';
import { REQUIRED_COLUMNS } from '../constants';

declare const Papa: any;

export const parseCSV = (file: File): Promise<HistoricalDraw[]> => {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: (results: any) => {
        if (results.errors.length) {
          return reject(new Error(`CSV Parsing Error: ${results.errors[0].message}`));
        }

        const headers = results.meta.fields;
        if (!REQUIRED_COLUMNS.every(col => headers.includes(col))) {
          return reject(new Error(`CSV file is missing one or more required columns. Required: ${REQUIRED_COLUMNS.join(', ')}`));
        }

        // Filter out any rows that might be null or have missing essential data
        const validData = results.data.filter((row: any) =>
            REQUIRED_COLUMNS.every(col => row[col] !== null && row[col] !== undefined)
        );

        resolve(validData as HistoricalDraw[]);
      },
      error: (error: Error) => {
        reject(error);
      },
    });
  });
};


export const generateRandomTicket = (): Ticket => {
    const whiteBalls = new Set<number>();
    while (whiteBalls.size < 5) {
        whiteBalls.add(Math.floor(Math.random() * 69) + 1);
    }
    const powerball = Math.floor(Math.random() * 26) + 1;
    return [Array.from(whiteBalls).sort((a, b) => a - b), powerball];
};

export const evaluateTicket = (ticket: Ticket, historicalDraw: HistoricalDraw): PrizeTier => {
    const [ticketWb, ticketPb] = ticket;

    const historicalWb = new Set([
        historicalDraw.wb1,
        historicalDraw.wb2,
        historicalDraw.wb3,
        historicalDraw.wb4,
        historicalDraw.wb5,
    ]);

    const matchedWbCount = ticketWb.filter(ball => historicalWb.has(ball)).length;
    const isPbMatch = ticketPb === historicalDraw.pb;

    if (matchedWbCount === 5 && isPbMatch) return "Grand Prize";
    if (matchedWbCount === 5 && !isPbMatch) return "$1 Million";
    if (matchedWbCount === 4 && isPbMatch) return "$50,000";
    if (matchedWbCount === 4 && !isPbMatch) return "$100 (4+0)";
    if (matchedWbCount === 3 && isPbMatch) return "$100 (3+1)";
    if (matchedWbCount === 3 && !isPbMatch) return "$7 (3+0)";
    if (matchedWbCount === 2 && isPbMatch) return "$7 (2+1)";
    if (matchedWbCount === 1 && isPbMatch) return "$4 (1+1)";
    if (matchedWbCount === 0 && isPbMatch) return "$4 (0+1)";

    return "No Win";
};

export const runBacktest = (dataframe: HistoricalDraw[]): BacktestResults => {
    const resultsTally: BacktestResults = {
        "Grand Prize": 0,
        "$1 Million": 0,
        "$50,000": 0,
        "$100 (4+0)": 0,
        "$100 (3+1)": 0,
        "$7 (3+0)": 0,
        "$7 (2+1)": 0,
        "$4 (1+1)": 0,
        "$4 (0+1)": 0,
        "No Win": 0,
    };

    for (const row of dataframe) {
        const ticket = generateRandomTicket();
        const prizeTier = evaluateTicket(ticket, row);
        resultsTally[prizeTier]++;
    }

    return resultsTally;
};
