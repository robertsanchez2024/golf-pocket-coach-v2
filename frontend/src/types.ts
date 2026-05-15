export interface SwingMetric {
    label: string;
    value: number;
    status: string;
    proRange: [number, number];
}

export interface SwingSession {
    id: number;
    date: string;
    club: string;
    metrics: SwingMetric[];
}